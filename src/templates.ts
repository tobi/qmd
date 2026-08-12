/**
 * Template-aware code retrieval engine for QMD.
 *
 * Supports generic web template engines including Blade, Twig, Smarty, and Latte.
 * Provides template language detection, structural breakpoint extraction,
 * and contextual metadata.
 */

import { extname } from "node:path";
import type { BreakPoint } from "./store.js";

export type SupportedTemplateEngine = "blade" | "twig" | "smarty" | "latte";

export interface TemplateContext {
  engine: SupportedTemplateEngine;
  extendsLayout?: string;
  sections?: string[];
  components?: string[];
}

export interface TemplateLanguageAdapter {
  engine: SupportedTemplateEngine;
  matches(filepath: string): boolean;
  getBreakPoints(content: string): BreakPoint[];
  getContext(content: string): TemplateContext;
}

// =============================================================================
// Blade Adapter
// =============================================================================

const BLADE_PATTERNS: { regex: RegExp; type: string; score: number }[] = [
  { regex: /@extends\s*\(/gi, type: "template:extends", score: 100 },
  { regex: /@section\s*\(/gi, type: "template:section", score: 90 },
  { regex: /@endsection\b/gi, type: "template:endsection", score: 70 },
  { regex: /@component\s*\(/gi, type: "template:component", score: 90 },
  { regex: /@endcomponent\b/gi, type: "template:endcomponent", score: 70 },
  { regex: /@push\s*\(/gi, type: "template:push", score: 80 },
  { regex: /@endpush\b/gi, type: "template:endpush", score: 70 },
  { regex: /@if\s*\(/gi, type: "template:if", score: 70 },
  { regex: /@elseif\s*\(/gi, type: "template:elseif", score: 60 },
  { regex: /@else\b/gi, type: "template:else", score: 60 },
  { regex: /@endif\b/gi, type: "template:endif", score: 60 },
  { regex: /@foreach\s*\(/gi, type: "template:foreach", score: 70 },
  { regex: /@endforeach\b/gi, type: "template:endforeach", score: 60 },
  { regex: /@for\s*\(/gi, type: "template:for", score: 70 },
  { regex: /@endfor\b/gi, type: "template:endfor", score: 60 },
  { regex: /@php\b/gi, type: "template:php", score: 80 },
  { regex: /@endphp\b/gi, type: "template:endphp", score: 60 },
  { regex: /<x-[\w.-]+/gi, type: "template:x-component", score: 85 },
  { regex: /<livewire:[\w.-]+/gi, type: "template:livewire-component", score: 85 },
];

export const bladeAdapter: TemplateLanguageAdapter = {
  engine: "blade",
  matches(filepath: string): boolean {
    const lower = filepath.toLowerCase();
    return lower.endsWith(".blade.php");
  },
  getBreakPoints(content: string): BreakPoint[] {
    const points: BreakPoint[] = [];
    const seen = new Set<number>();

    for (const pat of BLADE_PATTERNS) {
      pat.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pat.regex.exec(content)) !== null) {
        const pos = match.index;
        if (!seen.has(pos)) {
          seen.add(pos);
          points.push({ pos, score: pat.score, type: pat.type });
        }
      }
    }

    return points.sort((a, b) => a.pos - b.pos);
  },
  getContext(content: string): TemplateContext {
    const context: TemplateContext = { engine: "blade" };
    const extendsMatch = /@extends\s*\(\s*['"]([^'"]+)['"]\s*\)/i.exec(content);
    if (extendsMatch?.[1]) {
      context.extendsLayout = extendsMatch[1];
    }

    const sections: string[] = [];
    const secRegex = /@section\s*\(\s*['"]([^'"]+)['"]/gi;
    let sMatch: RegExpExecArray | null;
    while ((sMatch = secRegex.exec(content)) !== null) {
      if (sMatch[1] && !sections.includes(sMatch[1])) {
        sections.push(sMatch[1]);
      }
    }
    if (sections.length > 0) context.sections = sections;

    const components: string[] = [];
    const compRegex = /<(x-[\w.-]+|livewire:[\w.-]+)/gi;
    let cMatch: RegExpExecArray | null;
    while ((cMatch = compRegex.exec(content)) !== null) {
      if (cMatch[1] && !components.includes(cMatch[1])) {
        components.push(cMatch[1]);
      }
    }
    if (components.length > 0) context.components = components;

    return context;
  },
};

// =============================================================================
// Twig Adapter
// =============================================================================

const TWIG_PATTERNS: { regex: RegExp; type: string; score: number }[] = [
  { regex: /\{%\s*extends\b/gi, type: "template:extends", score: 100 },
  { regex: /\{%\s*block\b/gi, type: "template:block", score: 90 },
  { regex: /\{%\s*endblock\b/gi, type: "template:endblock", score: 70 },
  { regex: /\{%\s*include\b/gi, type: "template:include", score: 80 },
  { regex: /\{%\s*embed\b/gi, type: "template:embed", score: 90 },
  { regex: /\{%\s*endembed\b/gi, type: "template:endembed", score: 70 },
  { regex: /\{%\s*macro\b/gi, type: "template:macro", score: 90 },
  { regex: /\{%\s*endmacro\b/gi, type: "template:endmacro", score: 70 },
  { regex: /\{%\s*for\b/gi, type: "template:for", score: 70 },
  { regex: /\{%\s*endfor\b/gi, type: "template:endfor", score: 60 },
  { regex: /\{%\s*if\b/gi, type: "template:if", score: 70 },
  { regex: /\{%\s*endif\b/gi, type: "template:endif", score: 60 },
];

export const twigAdapter: TemplateLanguageAdapter = {
  engine: "twig",
  matches(filepath: string): boolean {
    const lower = filepath.toLowerCase();
    return lower.endsWith(".twig") || lower.endsWith(".html.twig");
  },
  getBreakPoints(content: string): BreakPoint[] {
    const points: BreakPoint[] = [];
    const seen = new Set<number>();

    for (const pat of TWIG_PATTERNS) {
      pat.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pat.regex.exec(content)) !== null) {
        const pos = match.index;
        if (!seen.has(pos)) {
          seen.add(pos);
          points.push({ pos, score: pat.score, type: pat.type });
        }
      }
    }

    return points.sort((a, b) => a.pos - b.pos);
  },
  getContext(content: string): TemplateContext {
    const context: TemplateContext = { engine: "twig" };
    const extendsMatch = /\{%\s*extends\s+['"]([^'"]+)['"]\s*%\}/i.exec(content);
    if (extendsMatch?.[1]) {
      context.extendsLayout = extendsMatch[1];
    }

    const blocks: string[] = [];
    const blkRegex = /\{%\s*block\s+([a-zA-Z0-9_]+)/gi;
    let bMatch: RegExpExecArray | null;
    while ((bMatch = blkRegex.exec(content)) !== null) {
      if (bMatch[1] && !blocks.includes(bMatch[1])) {
        blocks.push(bMatch[1]);
      }
    }
    if (blocks.length > 0) context.sections = blocks;

    return context;
  },
};

// =============================================================================
// Smarty Adapter
// =============================================================================

const SMARTY_PATTERNS: { regex: RegExp; type: string; score: number }[] = [
  { regex: /\{extends\b/gi, type: "template:extends", score: 100 },
  { regex: /\{block\b/gi, type: "template:block", score: 90 },
  { regex: /\{\/block\}/gi, type: "template:endblock", score: 70 },
  { regex: /\{include\b/gi, type: "template:include", score: 80 },
  { regex: /\{function\b/gi, type: "template:function", score: 90 },
  { regex: /\{\/function\}/gi, type: "template:endfunction", score: 70 },
  { regex: /\{if\b/gi, type: "template:if", score: 70 },
  { regex: /\{\/if\}/gi, type: "template:endif", score: 60 },
  { regex: /\{foreach\b/gi, type: "template:foreach", score: 70 },
  { regex: /\{\/foreach\}/gi, type: "template:endforeach", score: 60 },
];

export const smartyAdapter: TemplateLanguageAdapter = {
  engine: "smarty",
  matches(filepath: string): boolean {
    const lower = filepath.toLowerCase();
    return lower.endsWith(".tpl") || lower.endsWith(".smarty");
  },
  getBreakPoints(content: string): BreakPoint[] {
    const points: BreakPoint[] = [];
    const seen = new Set<number>();

    for (const pat of SMARTY_PATTERNS) {
      pat.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pat.regex.exec(content)) !== null) {
        const pos = match.index;
        if (!seen.has(pos)) {
          seen.add(pos);
          points.push({ pos, score: pat.score, type: pat.type });
        }
      }
    }

    return points.sort((a, b) => a.pos - b.pos);
  },
  getContext(content: string): TemplateContext {
    const context: TemplateContext = { engine: "smarty" };
    const extendsMatch = /\{extends\s+file=['"]([^'"]+)['"]\}/i.exec(content) || /\{extends\s+['"]([^'"]+)['"]\}/i.exec(content);
    if (extendsMatch?.[1]) {
      context.extendsLayout = extendsMatch[1];
    }

    const blocks: string[] = [];
    const blkRegex = /\{block\s+(?:name=)?['"]([^'"]+)['"]\}/gi;
    let bMatch: RegExpExecArray | null;
    while ((bMatch = blkRegex.exec(content)) !== null) {
      if (bMatch[1] && !blocks.includes(bMatch[1])) {
        blocks.push(bMatch[1]);
      }
    }
    if (blocks.length > 0) context.sections = blocks;

    return context;
  },
};

// =============================================================================
// Latte Adapter
// =============================================================================

const LATTE_PATTERNS: { regex: RegExp; type: string; score: number }[] = [
  { regex: /\{layout\b/gi, type: "template:layout", score: 100 },
  { regex: /\{block\b/gi, type: "template:block", score: 90 },
  { regex: /\{\/block\}/gi, type: "template:endblock", score: 70 },
  { regex: /\{include\b/gi, type: "template:include", score: 80 },
  { regex: /\{define\b/gi, type: "template:define", score: 90 },
  { regex: /\{\/define\}/gi, type: "template:enddefine", score: 70 },
  { regex: /\{if\b/gi, type: "template:if", score: 70 },
  { regex: /\{\/if\}/gi, type: "template:endif", score: 60 },
  { regex: /\{foreach\b/gi, type: "template:foreach", score: 70 },
  { regex: /\{\/foreach\}/gi, type: "template:endforeach", score: 60 },
];

export const latteAdapter: TemplateLanguageAdapter = {
  engine: "latte",
  matches(filepath: string): boolean {
    const lower = filepath.toLowerCase();
    return lower.endsWith(".latte");
  },
  getBreakPoints(content: string): BreakPoint[] {
    const points: BreakPoint[] = [];
    const seen = new Set<number>();

    for (const pat of LATTE_PATTERNS) {
      pat.regex.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pat.regex.exec(content)) !== null) {
        const pos = match.index;
        if (!seen.has(pos)) {
          seen.add(pos);
          points.push({ pos, score: pat.score, type: pat.type });
        }
      }
    }

    return points.sort((a, b) => a.pos - b.pos);
  },
  getContext(content: string): TemplateContext {
    const context: TemplateContext = { engine: "latte" };
    const layoutMatch = /\{layout\s+['"]([^'"]+)['"]\}/i.exec(content);
    if (layoutMatch?.[1]) {
      context.extendsLayout = layoutMatch[1];
    }

    const blocks: string[] = [];
    const blkRegex = /\{block\s+([a-zA-Z0-9_]+)/gi;
    let bMatch: RegExpExecArray | null;
    while ((bMatch = blkRegex.exec(content)) !== null) {
      if (bMatch[1] && !blocks.includes(bMatch[1])) {
        blocks.push(bMatch[1]);
      }
    }
    if (blocks.length > 0) context.sections = blocks;

    return context;
  },
};

// =============================================================================
// Template Engine Registry
// =============================================================================

const ADAPTERS: TemplateLanguageAdapter[] = [
  bladeAdapter,
  twigAdapter,
  smartyAdapter,
  latteAdapter,
];

/**
 * Detect template engine from filepath.
 * Priority rule: Check template adapters BEFORE standard extension mapping (.blade.php before .php).
 */
export function detectTemplateAdapter(filepath: string): TemplateLanguageAdapter | null {
  for (const adapter of ADAPTERS) {
    if (adapter.matches(filepath)) {
      return adapter;
    }
  }
  return null;
}

/**
 * Extract template break points and contextual header if applicable.
 */
export function getTemplateBreakPoints(content: string, filepath: string): BreakPoint[] {
  try {
    const adapter = detectTemplateAdapter(filepath);
    if (!adapter) return [];
    return adapter.getBreakPoints(content);
  } catch (err) {
    console.warn(`[qmd] Template parse failed for ${filepath}, falling back: ${err instanceof Error ? err.message : err}`);
    return [];
  }
}

/**
 * Get structural context for a template file.
 */
export function getTemplateContext(content: string, filepath: string): TemplateContext | null {
  try {
    const adapter = detectTemplateAdapter(filepath);
    if (!adapter) return null;
    return adapter.getContext(content);
  } catch {
    return null;
  }
}
