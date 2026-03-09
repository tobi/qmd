{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # SQLite with loadable extension support for sqlite-vec
        sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
          configureFlags = (old.configureFlags or []) ++ [
            "--enable-load-extension"
          ];
        });

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [
            pkgs.bun
            pkgs.makeWrapper
            pkgs.python3 # needed by node-gyp to compile better-sqlite3
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.darwin.cctools # provides libtool needed by node-gyp on macOS
          ];

          buildInputs = [ pkgs.sqlite ];

          dontConfigure = true;

          buildPhase = ''
            export HOME=$(mktemp -d)
            # Copy pre-downloaded deps, then fix shebangs and run install scripts
            cp -r ${bunDeps}/node_modules node_modules
            chmod -R u+w node_modules
            patchShebangs node_modules

            # Run lifecycle scripts (e.g. better-sqlite3 native compilation)
            # without allowing bun to reach the network.
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
            description = "On-device search engine for markdown notes, meeting transcripts, and knowledge bases";
            homepage = "https://github.com/tobi/qmd";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };

        # Fixed-output derivation: download deps only (no lifecycle scripts).
        # FODs get network access because the output is content-addressed.
        # --ignore-scripts avoids native compilation, which would introduce
        # Nix store references that FODs are not allowed to contain.
        bunDeps = pkgs.stdenv.mkDerivation {
          pname = "qmd-deps";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.bun ];

          dontConfigure = true;
          dontFixup = true;

          buildPhase = ''
            export HOME=$(mktemp -d)
            bun install --frozen-lockfile --ignore-scripts
          '';

          installPhase = ''
            mkdir -p $out
            cp -r node_modules $out/node_modules
          '';

          outputHashMode = "recursive";
          outputHashAlgo = "sha256";
          outputHash = "sha256-gfQmBdLdapDsZrG3QjoBvCNmzuTx4ek7l7Vi26z1Vbc=";
        };
      in
      {
        packages = {
          default = qmd;
          qmd = qmd;
        };

        apps.default = {
          type = "app";
          program = "${qmd}/bin/qmd";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.bun
            sqliteWithExtensions
          ];

          shellHook = ''
            export BREW_PREFIX="''${BREW_PREFIX:-${sqliteWithExtensions.out}}"
            echo "QMD development shell"
            echo "Run: bun src/qmd.ts <command>"
          '';
        };
      }
    );
}
