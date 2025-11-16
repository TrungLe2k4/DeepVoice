// background.js — DeepVoice Guard toggle per tab
// Mỗi tab Meet có trạng thái ON/OFF riêng.

const tabStates = new Map(); // tabId -> boolean (true = đang giám sát)

function isMeetUrl(url) {
  return typeof url === "string" && url.startsWith("https://meet.google.com/");
}

function setBadge(tabId, on) {
  chrome.action.setBadgeText({
    tabId,
    text: on ? "ON" : ""
  });
  chrome.action.setBadgeBackgroundColor({
    tabId,
    color: on ? "#e53935" : "#9e9e9e"
  });
}

// Khi click icon extension → toggle giám sát cho tab hiện tại
chrome.action.onClicked.addListener((tab) => {
  if (!tab || tab.id == null) return;
  const tabId = tab.id;

  if (!isMeetUrl(tab.url || "")) {
    // Không phải trang Meet → gửi thông tin nhẹ (nếu content.js có handle DV_INFO)
    chrome.tabs.sendMessage(
      tabId,
      {
        type: "DV_INFO",
        message: "Hãy mở Google Meet (https://meet.google.com) rồi bật lại DeepVoice Guard."
      },
      () => {
        // Bỏ qua lỗi nếu không có content script ở trang này
        void chrome.runtime.lastError;
      }
    );
    return;
  }

  // Đảo trạng thái theo tab
  const current = tabStates.get(tabId) === true;
  const next = !current;
  tabStates.set(tabId, next);
  setBadge(tabId, next);

  // Gửi DV_TOGGLE cho content.js (nơi sẽ gọi start()/stop())
  chrome.tabs.sendMessage(
    tabId,
    { type: "DV_TOGGLE" },
    () => {
      // Bỏ qua lỗi nếu content.js chưa được inject
      void chrome.runtime.lastError;
    }
  );
});

// Khi tab bị đóng → xoá trạng thái
chrome.tabs.onRemoved.addListener((tabId) => {
  tabStates.delete(tabId);
});

// Khi tab load xong 1 URL mới:
// - Nếu rời khỏi Meet → tắt badge + xoá trạng thái
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab && tab.url) {
    if (!isMeetUrl(tab.url)) {
      tabStates.delete(tabId);
      setBadge(tabId, false);
    }
  }
});

// Khi chuyển focus qua tab khác → cập nhật badge đúng trạng thái tab đó
chrome.tabs.onActivated.addListener(({ tabId }) => {
  const on = tabStates.get(tabId) === true;
  setBadge(tabId, on);
});

// Khi cài mới / reload extension
chrome.runtime.onInstalled.addListener(() => {
  chrome.action.setBadgeText({ text: "" });
  chrome.action.setBadgeBackgroundColor({ color: "#e53935" });
});
