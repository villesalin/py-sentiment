var port = chrome.runtime.connectNative("fi.tuni.sentiment");
port.onMessage.addListener(function (msg) {
	//forms a port to the backend
	console.log("Received" + msg);
	chrome.notifications.create("", {
		title: "Sentiment Analysis",
		message: `Your piece of text was ${msg}`,
		iconUrl: "/hi.png",
		type: "basic",
	});
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
		chrome.notifications.create("", {
			title: "Sentiment Analysis",
			message: `Analyzing...`,
			iconUrl: "/hi.png",
			type: "basic",
		});
		port.postMessage(clickData.selectionText);
	}
});

// chrome.windows.create({
// 	width: 350,
// 	height: 250,
// 	top: 200,
// 	left: 400,
// 	type: 'popup',
// 	url: 'hello.html',
// });
// console.log('Hello');
//});
// port.onMessage.addListener(function (msg) {
// 	console.log('Received' + msg);
// });
