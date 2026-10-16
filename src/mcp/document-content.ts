/** Text-first document response for MCP clients that cannot render embedded resources. */
export type DocumentContentInput = {
  uri: string;
  name: string;
  title?: string;
  text: string;
};

export type DocumentContentOptions = {
  exposeToUser?: boolean;
  confirmUserApprovedExposure?: true;
};

export function documentReadContent(doc: DocumentContentInput, options: DocumentContentOptions = {}) {
  if (!options.exposeToUser) {
    return { type: "text" as const, text: `# ${doc.name}\n\n${doc.text}` };
  }
  if (options.confirmUserApprovedExposure !== true) {
    throw new Error("User-visible QMD resource exposure requires explicit user approval.");
  }
  return {
    type: "resource" as const,
    resource: {
      uri: doc.uri,
      name: doc.name,
      title: doc.title,
      mimeType: "text/markdown" as const,
      text: doc.text,
    },
  };
}
