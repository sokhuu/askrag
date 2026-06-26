export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { question, chat_history } = req.body;

  const backendResponse = await fetch(`${process.env.RAG_BACKEND_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.RAG_API_KEY,
    },
    body: JSON.stringify({ question, chat_history: chat_history || [] }),
  });

  const data = await backendResponse.json();
  return res.status(backendResponse.status).json(data);
}
