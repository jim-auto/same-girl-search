"use client";

/* eslint-disable @next/next/no-img-element -- Images come from object URLs and the local API, not static Next assets. */
import { FormEvent, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Database, ExternalLink, ImageUp, Loader2, Search, Trash2, WifiOff } from "lucide-react";
import { deleteIndexedImage, getIndexStats, imageFileUrl, searchImage, SearchResponse } from "@/lib/api";

function bboxStyle(bbox: number[], imageWidth: number, imageHeight: number) {
  if (bbox.length < 4 || !imageWidth || !imageHeight) return {};
  const [x1, y1, x2, y2] = bbox;
  return {
    left: `${(x1 / imageWidth) * 100}%`,
    top: `${(y1 / imageHeight) * 100}%`,
    width: `${((x2 - x1) / imageWidth) * 100}%`,
    height: `${((y2 - y1) / imageHeight) * 100}%`
  };
}

export function SearchPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(10);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [indexCount, setIndexCount] = useState<number | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [adminMode, setAdminMode] = useState(false);
  const [showBbox, setShowBbox] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(() => Boolean(file) && !loading, [file, loading]);
  const isConnectionError = Boolean(error?.startsWith("API connection failed") || statsError?.startsWith("API connection failed"));

  useEffect(() => {
    let active = true;
    getIndexStats()
      .then((stats) => {
        if (!active) return;
        setIndexCount(stats.vectors);
        setStatsError(null);
      })
      .catch((err) => {
        if (!active) return;
        setStatsError(err instanceof Error ? err.message : "Could not load index stats");
      });
    return () => {
      active = false;
    };
  }, []);

  function onFile(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(nextFile ? URL.createObjectURL(nextFile) : null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await searchImage(file, topK));
      const stats = await getIndexStats();
      setIndexCount(stats.vectors);
      setStatsError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function refreshStats() {
    const stats = await getIndexStats();
    setIndexCount(stats.vectors);
    setStatsError(null);
  }

  async function removeResult(imageId: number) {
    setDeletingId(imageId);
    setError(null);
    try {
      await deleteIndexedImage(imageId);
      setResult((current) =>
        current
          ? {
              ...current,
              results: current.results
                .filter((item) => item.image.id !== imageId)
                .map((item, index) => ({ ...item, rank: index + 1 }))
            }
          : current
      );
      await refreshStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-2 border-b border-zinc-200 pb-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">same-girl-search</h1>
            <p className="max-w-3xl text-sm leading-6 text-zinc-600">
              ローカルに保存した画像インデックスから、顔 embedding の cosine similarity で候補プロフィールを検索します。
            </p>
          </div>
          <div className="inline-flex w-fit items-center gap-2 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-700">
            <Database className="h-4 w-4 text-zinc-500" aria-hidden="true" />
            <span>{indexCount === null ? "index: checking" : `index: ${indexCount} vectors`}</span>
          </div>
        </div>
        <div className="flex gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-900">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" aria-hidden="true" />
          <p>
            合意のない実人物画像の収集・追跡には使わないでください。デモと開発では synthetic、許諾済み、または mock
            data を使用します。
          </p>
        </div>
        {statsError ? (
          <div className="flex gap-2 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            <WifiOff className="h-4 w-4 flex-none" aria-hidden="true" />
            <p>{statsError}</p>
          </div>
        ) : null}
      </header>

      <main className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <form onSubmit={submit} className="flex flex-col gap-4 rounded-md border border-zinc-200 bg-white p-4">
          <label
            className="flex min-h-64 cursor-pointer flex-col items-center justify-center gap-3 rounded-md border border-dashed border-zinc-300 bg-zinc-50 p-4 text-center hover:bg-zinc-100"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              onFile(event.dataTransfer.files.item(0));
            }}
          >
            {preview ? (
              <img src={preview} alt="query preview" className="max-h-72 rounded-md object-contain" />
            ) : (
              <>
                <ImageUp className="h-10 w-10 text-zinc-500" aria-hidden="true" />
                <span className="text-sm font-medium text-zinc-800">画像をドロップまたは選択</span>
              </>
            )}
            <input
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={(event) => onFile(event.target.files?.item(0) ?? null)}
            />
          </label>

          <label className="flex flex-col gap-2 text-sm font-medium text-zinc-800">
            top-k
            <input
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="h-10 rounded-md border border-zinc-300 px-3"
            />
          </label>

          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-zinc-950 px-4 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Search className="h-4 w-4" aria-hidden="true" />}
            検索
          </button>

          {error ? (
            <div className="flex gap-2 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {isConnectionError ? <WifiOff className="h-4 w-4 flex-none" aria-hidden="true" /> : null}
              <p>{error}</p>
            </div>
          ) : null}
          {result ? (
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-700">
              query faces: {result.query.face_count} / det score: {result.query.det_score.toFixed(3)}
            </div>
          ) : null}

          <label className="flex items-start gap-3 rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-700">
            <input
              type="checkbox"
              checked={adminMode}
              onChange={(event) => setAdminMode(event.target.checked)}
              className="mt-1 h-4 w-4 rounded border-zinc-300"
            />
            <span>
              admin mode
              <span className="block text-xs leading-5 text-zinc-500">削除ボタンを表示します。削除後は FAISS index も再構築されます。</span>
            </span>
          </label>

          <label className="flex items-center gap-3 rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-700">
            <input
              type="checkbox"
              checked={showBbox}
              onChange={(event) => setShowBbox(event.target.checked)}
              className="h-4 w-4 rounded border-zinc-300"
            />
            <span>bbox overlay</span>
          </label>
        </form>

        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-zinc-950">候補</h2>
            <span className="text-sm text-zinc-500">{result ? `${result.results.length} results` : "no search yet"}</span>
          </div>

          {result?.results.length ? (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {result.results.map((item) => (
                <article key={`${item.rank}-${item.image.id}`} className="overflow-hidden rounded-md border border-zinc-200 bg-white">
                  <div className="flex aspect-[4/3] items-center justify-center bg-zinc-100 p-3">
                    <div
                      className="relative h-full max-h-full max-w-full"
                      style={{ aspectRatio: `${item.image.width || 1} / ${item.image.height || 1}` }}
                    >
                      <img src={imageFileUrl(item.image.id)} alt={item.image.image_path} className="h-full w-full object-contain" />
                      {showBbox ? (
                        <div
                          className="pointer-events-none absolute border-2 border-amber-400 shadow-[0_0_0_1px_rgba(0,0,0,0.45)]"
                          style={bboxStyle(item.image.bbox, item.image.width, item.image.height)}
                          aria-hidden="true"
                        />
                      ) : null}
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-zinc-900">#{item.rank}</span>
                      <span className="rounded-md bg-emerald-50 px-2 py-1 text-sm font-semibold text-emerald-700">
                        {item.score.toFixed(4)}
                      </span>
                    </div>
                    <dl className="grid grid-cols-[80px_1fr] gap-x-2 gap-y-1 text-xs text-zinc-600">
                      <dt>site</dt>
                      <dd className="truncate text-zinc-900">{item.image.source_site}</dd>
                      <dt>shop</dt>
                      <dd className="truncate text-zinc-900">{item.image.shop_name}</dd>
                      <dt>path</dt>
                      <dd className="truncate text-zinc-900" title={item.image.image_path}>
                        {item.image.image_path}
                      </dd>
                    </dl>
                    <div className="flex flex-wrap gap-2">
                      <a
                        className="inline-flex h-9 items-center gap-2 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-800 hover:bg-zinc-50"
                        href={imageFileUrl(item.image.id)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <ExternalLink className="h-4 w-4" aria-hidden="true" />
                        open
                      </a>
                      {item.image.profile_url ? (
                        <a
                          className="inline-flex h-9 items-center gap-2 rounded-md border border-blue-200 px-3 text-sm font-medium text-blue-700 hover:bg-blue-50"
                          href={item.image.profile_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <ExternalLink className="h-4 w-4" aria-hidden="true" />
                          profile
                        </a>
                      ) : null}
                      {adminMode ? (
                        <button
                          type="button"
                          disabled={deletingId === item.image.id}
                          onClick={() => removeResult(item.image.id)}
                          className="inline-flex h-9 items-center gap-2 rounded-md border border-red-200 px-3 text-sm font-medium text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {deletingId === item.image.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                          ) : (
                            <Trash2 className="h-4 w-4" aria-hidden="true" />
                          )}
                          delete
                        </button>
                      ) : null}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : result ? (
            <div className="flex min-h-80 items-center justify-center rounded-md border border-zinc-200 bg-white p-8 text-center text-sm text-zinc-500">
              {indexCount === 0
                ? "インデックスが空です。mock seed またはフォルダ indexing を実行してください。"
                : "検索は完了しましたが、候補はありませんでした。"}
            </div>
          ) : (
            <div className="flex min-h-80 items-center justify-center rounded-md border border-zinc-200 bg-white p-8 text-sm text-zinc-500">
              インデックス作成後、画像を検索すると候補が表示されます。
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
