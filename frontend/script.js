const form = document.getElementById("watermark-form");
const fileInput = document.getElementById("pdf-file");
const keywordsInput = document.getElementById("keywords");
const modeSelect = document.getElementById("mode");
const submitBtn = document.getElementById("submit-btn");
const resultSection = document.getElementById("result");
const downloadLink = document.getElementById("download-link");
const statusBox = document.getElementById("status");
const progressSection = document.getElementById("progress-section");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressPercentage = document.getElementById("progress-percentage");
const logList = document.getElementById("log-list");

const supportsWebSocket = typeof window !== "undefined" && "WebSocket" in window;
const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per chunk

let currentProgress = 0;

let activeSocket = null;
let socketFinished = false;
let fallbackAttempted = false;
let previousDownloadUrl = null;

const toggleLoadingState = (loading) => {
  submitBtn.disabled = loading;
  submitBtn.textContent = loading ? "处理中..." : "开始去除";
};

const showStatus = (message, isError = false) => {
  statusBox.textContent = message;
  statusBox.classList.toggle("hidden", false);
  statusBox.style.background = isError
    ? "rgba(248, 113, 113, 0.15)"
    : "rgba(251, 191, 36, 0.15)";
  statusBox.style.color = isError ? "#991b1b" : "#92400e";
};

const hideStatus = () => {
  statusBox.classList.add("hidden");
  statusBox.textContent = "";
};

const updateProgress = (value) => {
  const bounded = Math.max(0, Math.min(100, value));
  currentProgress = Math.max(currentProgress, bounded);
  progressBarFill.style.width = `${currentProgress}%`;
  progressPercentage.textContent = `${Math.round(currentProgress)}%`;
};

const appendLog = (message) => {
  if (!message) {
    return;
  }
  const time = new Date().toLocaleTimeString();
  const item = document.createElement("li");
  item.textContent = `[${time}] ${message}`;
  logList.appendChild(item);
  if (logList.children.length > 60) {
    logList.removeChild(logList.firstChild);
  }
  logList.scrollTop = logList.scrollHeight;
};

const resetDownloadLink = () => {
  if (previousDownloadUrl) {
    URL.revokeObjectURL(previousDownloadUrl);
    previousDownloadUrl = null;
  }
  downloadLink.href = "#";
  downloadLink.removeAttribute("download");
  resultSection.classList.add("hidden");
};

const prepareProgress = () => {
  progressSection.classList.remove("hidden");
  logList.innerHTML = "";
  currentProgress = 0;
  updateProgress(0);
};

const closeActiveSocket = () => {
  if (!activeSocket) {
    return;
  }
  try {
    activeSocket.close(1000, "client_restart");
  } catch (error) {
    console.warn("Failed to close previous socket", error);
  }
  activeSocket = null;
};

const base64ToBlob = (base64, mimeType) => {
  const binary = window.atob(base64);
  const size = binary.length;
  const bytes = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
};

const arrayBufferToBase64 = (buffer) => {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkLength = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkLength) {
    const slice = bytes.subarray(offset, offset + chunkLength);
    binary += String.fromCharCode.apply(null, slice);
  }
  return window.btoa(binary);
};

const processViaHttp = async (file, keywords, mode) => {
  appendLog("使用 HTTP 轮询模式处理文件...");
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));

  try {
    const initResponse = await fetch("/api/upload/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileName: file.name,
        totalChunks,
        mode,
      }),
    });

    if (!initResponse.ok) {
      const payload = await initResponse.json().catch(() => ({}));
      throw new Error(payload.detail || "初始化上传会话失败");
    }

    const { uploadId } = await initResponse.json();
    appendLog(`已创建上传会话 (${uploadId})，开始上传分片...`);

    for (let index = 0; index < totalChunks; index += 1) {
      const chunkStart = index * CHUNK_SIZE;
      const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, file.size);
      const chunkBlob = file.slice(chunkStart, chunkEnd);

      const formData = new FormData();
      formData.append("upload_id", uploadId);
      formData.append("chunk_index", index.toString());
      formData.append("total_chunks", totalChunks.toString());
      formData.append("chunk", chunkBlob, `${file.name}.part${index}`);

      const chunkResponse = await fetch("/api/upload/chunk", {
        method: "POST",
        body: formData,
      });

      if (!chunkResponse.ok) {
        const payload = await chunkResponse.json().catch(() => ({}));
        throw new Error(payload.detail || `上传分片 ${index + 1} 失败`);
      }

      const uploadProgress = ((index + 1) / totalChunks) * 50;
      updateProgress(uploadProgress);
      appendLog(`分片 ${index + 1}/${totalChunks} 上传完成`);
    }

    appendLog("全部分片已上传，后台正在处理文件...");
    updateProgress(55);

    const finalizeForm = new FormData();
    finalizeForm.append("upload_id", uploadId);
    if (keywords) {
      finalizeForm.append("keywords", keywords);
    }
    finalizeForm.append("mode", mode);

    const response = await fetch("/api/remove-watermark", {
      method: "POST",
      body: finalizeForm,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "处理失败，请稍后再试");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    previousDownloadUrl = url;
    downloadLink.href = url;
    downloadLink.download = `cleaned-${file.name}`;

    appendLog("文件处理完成，已生成下载链接。");
    updateProgress(100);
    resultSection.classList.remove("hidden");
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : "处理失败，请稍后再试";
    showStatus(message, true);
    appendLog(message);
  } finally {
    toggleLoadingState(false);
  }
};

const fallbackToHttp = (file, keywords, mode, reason) => {
  if (fallbackAttempted) {
    return;
  }
  fallbackAttempted = true;
  socketFinished = true;
  if (reason) {
    appendLog(reason);
  } else {
    appendLog("切换到 HTTP 轮询模式继续处理...");
  }
  closeActiveSocket();
  processViaHttp(file, keywords, mode);
};

const processViaWebSocket = (file, keywords, mode) => {
  appendLog("正在建立与服务器的 WebSocket 连接...");
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socketUrl = `${protocol}://${window.location.host}/ws/remove-watermark`;
  const socket = new WebSocket(socketUrl);
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
  activeSocket = socket;
  socketFinished = false;

  const sendChunks = async () => {
    for (let index = 0; index < totalChunks; index += 1) {
      const chunkStart = index * CHUNK_SIZE;
      const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, file.size);
      const chunkBlob = file.slice(chunkStart, chunkEnd);

      const buffer = await chunkBlob.arrayBuffer();
      const base64Data = arrayBufferToBase64(buffer);

      socket.send(
        JSON.stringify({
          type: "chunk",
          index,
          data: base64Data,
        })
      );

      const uploadProgress = ((index + 1) / totalChunks) * 25;
      updateProgress(uploadProgress);
      appendLog(`已发送分片 ${index + 1}/${totalChunks}`);
    }

    socket.send(JSON.stringify({ type: "end" }));
    appendLog("全部分片已发送，等待服务器处理...");
  };

  socket.addEventListener("open", async () => {
    try {
      appendLog("连接成功，初始化上传会话...");
      socket.send(
        JSON.stringify({
          type: "start",
          fileName: file.name,
          keywords,
          totalChunks,
          mode,
        })
      );
      await sendChunks();
    } catch (error) {
      console.error(error);
      fallbackToHttp(file, keywords, mode, "分片上传失败，切换到 HTTP 模式继续处理。");
      return;
    }
  });

  socket.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      appendLog("收到无法解析的服务器消息");
      return;
    }

    if (payload.type === "status") {
      if (typeof payload.progress === "number") {
        updateProgress(payload.progress);
      }
      if (payload.message) {
        appendLog(payload.message);
      }
    } else if (payload.type === "complete") {
      socketFinished = true;
      updateProgress(100);
      if (payload.message) {
        appendLog(payload.message);
      }
      if (payload.fileData) {
        const blob = base64ToBlob(payload.fileData, "application/pdf");
        previousDownloadUrl = URL.createObjectURL(blob);
        downloadLink.href = previousDownloadUrl;
        downloadLink.download = payload.fileName || `cleaned-${file.name}`;
        resultSection.classList.remove("hidden");
      }
      toggleLoadingState(false);
      socket.close(1000, "completed");
    } else if (payload.type === "error") {
      socketFinished = true;
      toggleLoadingState(false);
      showStatus(payload.message || "处理失败", true);
      appendLog(payload.message || "处理失败");
      socket.close(1011, "error");
    } else {
      appendLog("收到未知类型的服务器消息");
    }
  });

  socket.addEventListener("close", () => {
    if (activeSocket === socket) {
      activeSocket = null;
    }
    if (!socketFinished) {
      if (!fallbackAttempted) {
        fallbackToHttp(file, keywords, mode, "WebSocket 连接已关闭，切换到 HTTP 模式继续处理。");
      } else {
        appendLog("WebSocket 连接已关闭，已切换为 HTTP 模式处理。");
      }
      return;
    }
    appendLog("WebSocket 连接已关闭。");
  });

  socket.addEventListener("error", () => {
    if (!fallbackAttempted) {
      fallbackToHttp(file, keywords, mode, "WebSocket 连接出现错误，切换到 HTTP 模式继续处理。");
    } else if (!statusBox.textContent) {
      showStatus("WebSocket 连接出现错误", true);
    }
    appendLog("WebSocket 连接出现错误");
  });
};

form.addEventListener("submit", (event) => {
  event.preventDefault();
  hideStatus();
  resetDownloadLink();
  prepareProgress();

  const file = fileInput.files?.[0];
  if (!file) {
    showStatus("请先选择一个 PDF 文件", true);
    progressSection.classList.add("hidden");
    return;
  }

  const keywords = keywordsInput.value.trim();
  const mode = modeSelect.value;
  appendLog(`选择的处理模式：${mode}`);

  toggleLoadingState(true);
  fallbackAttempted = false;
  socketFinished = false;
  closeActiveSocket();

  if (!supportsWebSocket) {
    fallbackAttempted = true;
    appendLog("当前环境不支持 WebSocket，使用 HTTP 轮询模式。");
    processViaHttp(file, keywords, mode);
    return;
  }

  processViaWebSocket(file, keywords, mode);
});
