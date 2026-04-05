export type EmbeddedSkillFile = {
    relativePath: string;
    content: string;
};
export declare function getEmbeddedQmdSkillFiles(): EmbeddedSkillFile[];
export declare function getEmbeddedQmdSkillContent(): string;
