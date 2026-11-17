const { contextBridge, ipcRenderer } = require("electron");
const channels = ["library", "service", "project", "sample", "take", "export"];
contextBridge.exposeInMainWorld("studio", Object.fromEntries(
  channels.map(channel => [channel, (...args) => ipcRenderer.invoke(channel, ...args)]),
));
