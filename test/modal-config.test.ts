/**
 * Unit tests for modal config keys in CollectionConfig.
 *
 * Tests that modal.inference, modal.gpu, and modal.scaledown_window
 * can be read/written via loadConfig/saveConfig, and that defaults
 * are applied correctly by getModalConfig/setModalConfig helpers.
 */

import { describe, test, expect } from "vitest";
import {
  loadConfig,
  saveConfig,
  setConfigSource,
  getModalConfig,
  setModalConfig,
} from "../src/collections.js";
import type { CollectionConfig } from "../src/collections.js";

/** Helper: run fn with an isolated in-memory config, then reset. */
function withInlineConfig(fn: () => void): void {
  setConfigSource({ config: { collections: {} } });
  try {
    fn();
  } finally {
    setConfigSource(undefined);
  }
}

describe("modal config keys", () => {
  test("loadConfig returns no modal block when none is set", () => {
    withInlineConfig(() => {
      const config = loadConfig();
      expect(config.modal).toBeUndefined();
    });
  });

  test("saveConfig persists modal block and loadConfig reads it back", () => {
    withInlineConfig(() => {
      const config: CollectionConfig = {
        collections: {},
        modal: {
          inference: true,
          gpu: "A10G",
          scaledown_window: 30,
        },
      };
      saveConfig(config);

      const loaded = loadConfig();
      expect(loaded.modal).toEqual({
        inference: true,
        gpu: "A10G",
        scaledown_window: 30,
      });
    });
  });

  test("round-trip preserves partial modal config", () => {
    withInlineConfig(() => {
      const config: CollectionConfig = {
        collections: {},
        modal: {
          inference: true,
        },
      };
      saveConfig(config);

      const loaded = loadConfig();
      expect(loaded.modal?.inference).toBe(true);
      expect(loaded.modal?.gpu).toBeUndefined();
      expect(loaded.modal?.scaledown_window).toBeUndefined();
    });
  });
});

describe("getModalConfig", () => {
  test("returns defaults when no modal config is set", () => {
    withInlineConfig(() => {
      const modal = getModalConfig();
      expect(modal).toEqual({
        inference: false,
        gpu: "T4",
        scaledown_window: 15,
      });
    });
  });

  test("returns stored values merged with defaults", () => {
    withInlineConfig(() => {
      saveConfig({
        collections: {},
        modal: { inference: true, gpu: "A10G" },
      });

      const modal = getModalConfig();
      expect(modal).toEqual({
        inference: true,
        gpu: "A10G",
        scaledown_window: 15,
      });
    });
  });

  test("returns all stored values when fully specified", () => {
    withInlineConfig(() => {
      saveConfig({
        collections: {},
        modal: { inference: true, gpu: "L4", scaledown_window: 60 },
      });

      const modal = getModalConfig();
      expect(modal).toEqual({
        inference: true,
        gpu: "L4",
        scaledown_window: 60,
      });
    });
  });
});

describe("setModalConfig", () => {
  test("sets modal config on empty config", () => {
    withInlineConfig(() => {
      setModalConfig({ inference: true });

      const config = loadConfig();
      expect(config.modal?.inference).toBe(true);
    });
  });

  test("merges partial update into existing modal config", () => {
    withInlineConfig(() => {
      saveConfig({
        collections: {},
        modal: { inference: true, gpu: "T4", scaledown_window: 15 },
      });

      setModalConfig({ gpu: "A10G" });

      const config = loadConfig();
      expect(config.modal).toEqual({
        inference: true,
        gpu: "A10G",
        scaledown_window: 15,
      });
    });
  });

  test("preserves existing collections when setting modal config", () => {
    withInlineConfig(() => {
      saveConfig({
        collections: {
          notes: { path: "/tmp/notes", pattern: "**/*.md" },
        },
      });

      setModalConfig({ inference: true });

      const config = loadConfig();
      expect(config.collections.notes).toBeDefined();
      expect(config.modal?.inference).toBe(true);
    });
  });
});
