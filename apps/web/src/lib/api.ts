export type IndexedImage = {
  id: number;
  image_path: string;
  source_site: string;
  shop_name: string;
  profile_url: string | null;
  bbox: number[];
  det_score: number;
  face_count: number;
  width: number;
  height: number;
};

export type SearchResult = {
  rank: number;
  score: number;
  image: IndexedImage;
};

export type SearchResponse = {
  query: {
    bbox: number[];
    det_score: number;
    face_count: number;
  };
  results: SearchResult[];
};

export type IndexStats = {
  vectors: number;
};

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseError(response: Response, fallback: string): Promise<Error> {
  const detail = await response.json().catch(() => ({ detail: response.statusText }));
  return new Error(typeof detail.detail === "string" ? detail.detail : fallback);
}

export async function getIndexStats(): Promise<IndexStats> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/index/stats`);
  } catch (error) {
    throw new Error(error instanceof Error ? `API connection failed: ${error.message}` : "API connection failed");
  }
  if (!response.ok) {
    throw await parseError(response, "Could not load index stats");
  }
  return response.json();
}

export async function searchImage(file: File, topK: number): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/search?top_k=${topK}`, {
      method: "POST",
      body: form
    });
  } catch (error) {
    throw new Error(error instanceof Error ? `API connection failed: ${error.message}` : "API connection failed");
  }
  if (!response.ok) {
    throw await parseError(response, "Search failed");
  }
  return response.json();
}

export async function deleteIndexedImage(id: number): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/images/${id}`, { method: "DELETE" });
  } catch (error) {
    throw new Error(error instanceof Error ? `API connection failed: ${error.message}` : "API connection failed");
  }
  if (!response.ok) {
    throw await parseError(response, "Delete failed");
  }
}

export function imageFileUrl(id: number): string {
  return `${API_BASE_URL}/images/${id}/file`;
}
