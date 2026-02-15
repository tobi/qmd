{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      imports = [ inputs.treefmt-nix.flakeModule ];

      perSystem =
        {
          pkgs,
          self',
          ...
        }:
        let
          sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
            configureFlags = (old.configureFlags or [ ]) ++ [ "--enable-load-extension" ];
          });
        in
        {
          packages = {
            default = self'.packages.qmd;
            qmd = pkgs.callPackage ./nix/package.nix {
              inherit (pkgs) vulkan-loader autoAddDriverRunpath cudaPackages;
            };
          };

          devShells.default = pkgs.mkShell {
            buildInputs = [
              pkgs.bun
              pkgs.nodejs # for npm (lockfile generation)
              pkgs.jq
              sqliteWithExtensions
            ];
            shellHook = ''
              export BREW_PREFIX="''${BREW_PREFIX:-${sqliteWithExtensions.out}}"
            '';
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              nixfmt.enable = true;
            };
          };
        };
    };
}
