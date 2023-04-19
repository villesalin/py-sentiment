let contextMenuItem = {
    "id": "sentiment",
    "title": "Sentiment",
    "contexts": ["selection"]
    };
chrome.runtime.onInstalled.addListener(() =>
    chrome.contextMenus.create(contextMenuItem)
);
chrome.contextMenus.onClicked.addListener((info, tab) => {
    chrome.windows.create({
    width: 350,
    height: 250,
    top: 200,
    left: 400,
    type: "popup",
    url: "hello.html"
    });
    console.log("Hello");
});