# Nix Build

This directory contains the Nix build configuration for QMD.

This project uses [bun2nix](https://nix-community.github.io/bun2nix/) to convert Bun dependencies into Nix derivations. This allows building without network access during the build phase by prefetching all npm packages into the Nix store.

## Development

Run `nix develop` to enter the dev shell created from the flake.

## Building

```sh
nix build
```

## Maintenance

### Updating Dependencies

When `bun.lock` changes, a GitHub Action automatically updates `bun.nix` and creates a PR.

For local updates, run:

```sh
bunx bun2nix -o nix/bun.nix
```

**Note**: The updated `nix/bun.nix` will be committed in the merge of the auto-generated PR.

### How It Works

- `flake.nix` - Main Nix flake configuration
- `bun.nix` - Generated Nix expression containing all npm dependencies

### Updating the Flake Inputs

```sh
nix flake update
```

# Installation

### Via Flakes (Recommended)

Add QMD as a flake input in your `flake.nix`:

```nix
{
  inputs.qmd.url = "github:tobi/qmd";
}
```

Then use it in your packages:

```nix
{
  packages = {
    qmd = inputs.qmd.packages.${system}.qmd;
  };
}
```
