export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.RAG_BACKEND_URL) {
    console.error("RAG_BACKEND_URL is not set");
    return res.status(500).json({ error: "Backend is not configured." });
  }

  const { event_id, context } = req.body;

  let backendResponse;
  try {
    backendResponse = await fetch(`${process.env.RAG_BACKEND_URL}/event`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": process.env.RAG_API_KEY,
      },
      body: JSON.stringify({ event_id, context: context || "" }),
    });
  } catch (err) {
    console.error("Failed to reach RAG backend:", err);
    return res.status(502).json({ error: "Could not reach the backend server." });
  }

  let data;
  try {
    data = await backendResponse.json();
  } catch (err) {
    const text = await backendResponse.text().catch(() => "");
    console.error("Backend returned non-JSON response:", backendResponse.status, text.slice(0, 500));
    return res.status(502).json({ error: "Backend returned an unexpected response." });
  }

  return res.status(backendResponse.status).json(data);
}
