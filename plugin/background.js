var port = chrome.runtime.connectNative("fi.tuni.sentiment");
port.onMessage.addListener(function (msg) {
	//forms a port to the backend
	console.log("Received" + msg);
});
let contextMenuItem = {
	"id": "sentiment",
	"title": "Sentiment",
	"contexts": ["selection"],
};
//creates the context menu
chrome.runtime.onInstalled.addListener(() => chrome.contextMenus.create(contextMenuItem));
//the listener checks that the right button is clicked and that there is text selected before sending the text to the backend
chrome.contextMenus.onClicked.addListener(function (clickData) {
	if (clickData.menuItemId == "sentiment" && clickData.selectionText) {
		port.postMessage(clickData.selectionText);
	}
});

// chrome.windows.create({
// 	width: 350,
// 	height: 250,
// 	top: 200,
// 	left: 400,
// 	type: "popup",
// 	url: "hello.html",
// });
// console.log("Hello");
//});
// port.onMessage.addListener(function (msg) {
// 	console.log("Received" + msg);
// });
