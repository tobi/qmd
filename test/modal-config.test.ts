/**
 * Unit tests for modal config keys in CollectionConfig.
 *
 * Tests that modal.inference, modal.gpu, and modal.scaledown_window
 * can be read/written via loadConfig/saveConfig, and that defaults
 * are applied correctly by getModalConfig/setModalConfig helpers.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import {
  loadConfig,
  saveConfig,
  setConfigSource,
  getModalConfig,
  setModalConfig,
} from "../src/collections.js";
import type { CollectionConfig } from "../src/collections.js";

beforeEach(() => {
  // Use in-memory config for each test
  setConfigSource({ config: { collections: {} } });
});

afterEach(() => {
  setConfigSource(undefined);
});

describe("modal config keys", () => {
  test("loadConfig returns no modal block when none is set", () => {
    const config = loadConfig();
    expect(config.modal).toBeUndefined();
  });

  test("saveConfig persists modal block and loadConfig reads it back", () => {
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

  test("round-trip preserves partial modal config", () => {
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

describe("getModalConfig", () => {
  test("returns defaults when no modal config is set", () => {
    const modal = getModalConfig();
    expect(modal).toEqual({
      inference: false,
      gpu: "T4",
      scaledown_window: 15,
    });
  });

  test("returns stored values merged with defaults", () => {
    saveConfig({
      collections: {},
      modal: { inference: true, gpu: "A10G" },
    });

    const modal = getModalConfig();
    expect(modal).toEqual({
      inference: true,
      gpu: "A10G",
      scaledown_window: 15, // default applied
    });
  });

  test("returns all stored values when fully specified", () => {
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

describe("setModalConfig", () => {
  test("sets modal config on empty config", () => {
    setModalConfig({ inference: true });

    const config = loadConfig();
    expect(config.modal?.inference).toBe(true);
  });

  test("merges partial update into existing modal config", () => {
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

  test("preserves existing collections when setting modal config", () => {
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
