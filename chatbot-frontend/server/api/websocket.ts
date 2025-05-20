export default defineWebSocketHandler({
  open(peer) {
    console.log("[ws] open");
    peer.send("Hallo 👋️ Ich bin Neo, dein Abfall-Assistent. Wie kann ich dir heute helfen? ")
  },

  message(peer, message) {
    console.log("[ws] message", message);
      peer.send("pong"+ Math.random());
  },

  close(peer, event) {
    console.log("[ws] close", event);
  },

  error(peer, error) {
    console.log("[ws] error", error);
  },
});
