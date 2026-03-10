{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # SQLite with loadable extension support for sqlite-vec
        sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
          configureFlags = (old.configureFlags or []) ++ [
            "--enable-load-extension"
          ];
        });

        # Rust build (new)
        qmd-rs = pkgs.rustPlatform.buildRustPackage {
          pname = "qmd";
          version = "1.1.5";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          meta = with pkgs.lib; {
            description = "On-device search engine for markdown notes, meeting transcripts, and knowledge bases";
            homepage = "https://github.com/tobi/qmd";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };

        # TypeScript build (legacy, kept for compatibility)
        qmd-ts = pkgs.stdenv.mkDerivation {
          pname = "qmd-ts";
          version = "1.1.5";

          src = ./.;

          nativeBuildInputs = [
            pkgs.bun
            pkgs.makeWrapper
            pkgs.python3  # needed by node-gyp to compile better-sqlite3
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.darwin.cctools  # provides libtool needed by node-gyp on macOS
          ];

          buildInputs = [ pkgs.sqlite ];

          buildPhase = ''
            export HOME=$(mktemp -d)
            bun install --frozen-lockfile
          '';

          installPhase = ''
            mkdir -p $out/lib/qmd
            mkdir -p $out/bin

            cp -r node_modules $out/lib/qmd/
            cp -r src $out/lib/qmd/
            cp package.json $out/lib/qmd/

            makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
              --add-flags "$out/lib/qmd/src/qmd.ts" \
              --set DYLD_LIBRARY_PATH "${pkgs.sqlite.out}/lib" \
              --set LD_LIBRARY_PATH "${pkgs.sqlite.out}/lib"
          '';

          meta = with pkgs.lib; {
            description = "On-device search engine for markdown notes (TypeScript version)";
            homepage = "https://github.com/tobi/qmd";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
      in
      {
        packages = {
          default = qmd-rs;
          qmd = qmd-rs;
          qmd-ts = qmd-ts;
        };

        apps.default = {
          type = "app";
          program = "${qmd-rs}/bin/qmd-rs";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.bun
            pkgs.cargo
            pkgs.rustc
            pkgs.rust-analyzer
            pkgs.clippy
            sqliteWithExtensions
          ];

          shellHook = ''
            export BREW_PREFIX="''${BREW_PREFIX:-${sqliteWithExtensions.out}}"
            echo "QMD development shell (Rust + TypeScript)"
            echo "Rust:       cargo build"
            echo "TypeScript: bun src/qmd.ts <command>"
          '';
        };
      }
    );
}
