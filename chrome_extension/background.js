// background.js (MV3, module)
chrome.runtime.onInstalled.addListener(() => {
  chrome.action.setBadgeText({ text: "OFF" });
  chrome.action.setBadgeBackgroundColor({ color: "#777" });
});

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab || !tab.id) return;
  const curr = await chrome.action.getBadgeText({tabId: tab.id});
  const nextOn = curr !== "ON";
  await chrome.action.setBadgeText({ tabId: tab.id, text: nextOn ? "ON" : "OFF" });
  await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: nextOn ? "#0a0" : "#777" });
  // Notify content script to start/stop analysis
  await chrome.tabs.sendMessage(tab.id, { type: nextOn ? "DV_START" : "DV_STOP" }).catch(()=>{});
});
