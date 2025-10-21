const form = document.getElementById("watermark-form");
const fileInput = document.getElementById("pdf-file");
const keywordsInput = document.getElementById("keywords");
const submitBtn = document.getElementById("submit-btn");
const resultSection = document.getElementById("result");
const downloadLink = document.getElementById("download-link");
const statusBox = document.getElementById("status");

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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideStatus();
  resultSection.classList.add("hidden");

  const file = fileInput.files?.[0];
  if (!file) {
    showStatus("请先选择一个 PDF 文件", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  const keywords = keywordsInput.value.trim();
  if (keywords.length > 0) {
    formData.append("keywords", keywords);
  }

  toggleLoadingState(true);

  try {
    const response = await fetch("/api/remove-watermark", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const message = await response.json().catch(() => ({}));
      throw new Error(message.detail || "处理失败，请稍后再试");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    downloadLink.href = url;
    downloadLink.download = `cleaned-${file.name}`;
    resultSection.classList.remove("hidden");
  } catch (error) {
    console.error(error);
    showStatus(error.message || "发生未知错误", true);
  } finally {
    toggleLoadingState(false);
  }
});
