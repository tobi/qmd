export async function handlePullCommand(
  values: Record<string, unknown>,
  deps: {
    pull: (refresh: boolean) => Promise<void>;
  }
): Promise<void> {
  const refresh = values.refresh === undefined ? false : Boolean(values.refresh);
  await deps.pull(refresh);
}

export function handleCleanupCommand(
  deps: {
    cleanup: () => void;
  }
): void {
  deps.cleanup();
}

export function handleStatusCommand(
  deps: {
    showStatus: () => void;
  }
): void {
  deps.showStatus();
}

export async function handleUpdateCommand(
  values: Record<string, unknown>,
  deps: {
    updateCollections: (allowRun: boolean) => Promise<void>;
  }
): Promise<void> {
  const allowRun = values["allow-run"] === undefined ? false : Boolean(values["allow-run"]);
  await deps.updateCollections(allowRun);
}

export async function handleEmbedCommand(
  values: Record<string, unknown>,
  deps: {
    vectorIndex: (model: string, force: boolean) => Promise<void>;
    defaultModel: string;
  }
): Promise<void> {
  await deps.vectorIndex(deps.defaultModel, !!values.force);
}

export async function handleMcpCommand(
  deps: {
    startMcpServer: () => Promise<void>;
  }
): Promise<void> {
  await deps.startMcpServer();
}

export async function handleDoctorCommand(
  deps: {
    doctor: (bench?: boolean) => Promise<void>;
  },
  bench = false
): Promise<void> {
  await deps.doctor(bench);
}
