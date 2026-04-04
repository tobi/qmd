export async function handleVSearchCommand<TOpts extends { minScore: number }>(
  query: string,
  values: Record<string, unknown>,
  opts: TOpts,
  deps: {
    vectorSearch: (query: string, opts: TOpts) => Promise<void>;
  }
): Promise<void> {
  if (!query) {
    console.error("Usage: qmd vsearch [options] <query>");
    process.exit(1);
  }
  if (!values["min-score"]) {
    opts.minScore = 0.3;
  }
  await deps.vectorSearch(query, opts);
}

export function handleSearchCommand<TOpts>(
  query: string,
  opts: TOpts,
  deps: {
    search: (query: string, opts: TOpts) => void;
  }
): void {
  if (!query) {
    console.error("Usage: qmd search [options] <query>");
    process.exit(1);
  }
  deps.search(query, opts);
}

export async function handleQueryCommand<TOpts>(
  query: string,
  opts: TOpts,
  deps: {
    querySearch: (query: string, opts: TOpts) => Promise<void>;
  }
): Promise<void> {
  if (!query) {
    console.error("Usage: qmd query [options] <query>");
    process.exit(1);
  }
  await deps.querySearch(query, opts);
}
