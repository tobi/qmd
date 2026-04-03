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

        nodeModulesHashes = {
          x86_64-linux = "sha256-Hymzuiid76j0LbDRACYlRQ2UVxQp7t9xg4nH37l0Keg=";
          aarch64-darwin = "sha256-/9kp5mNrI7hVR137DRpSuZHnl1RL/wFu2hKyzXW66TU=";

          # Populate these on first build for additional hosts if/when needed.
          aarch64-linux = pkgs.lib.fakeHash;
          x86_64-darwin = pkgs.lib.fakeHash;
        };

        nodeModules = pkgs.stdenvNoCC.mkDerivation {
          pname = "qmd-node-modules";
          version = "1.0.0";

          src = ./.;

          impureEnvVars = pkgs.lib.fetchers.proxyImpureEnvVars ++ [
            "GIT_PROXY_COMMAND"
            "SOCKS_SERVER"
          ];

          nativeBuildInputs = [
            pkgs.bun
          ];

          dontConfigure = true;

          buildPhase = ''
            export HOME=$(mktemp -d)

            bun install \
              --backend copyfile \
              --frozen-lockfile \
              --ignore-scripts \
              --no-progress \
              --production
          '';

          installPhase = ''
            mkdir -p $out
            cp -R node_modules $out/
          '';

          dontFixup = true;

          outputHash = nodeModulesHashes.${system};
          outputHashAlgo = "sha256";
          outputHashMode = "recursive";
        };

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [
            pkgs.bun
            pkgs.makeWrapper
            pkgs.nodejs
            pkgs.node-gyp
            pkgs.python3  # needed by node-gyp to compile better-sqlite3
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.darwin.cctools  # provides libtool needed by node-gyp on macOS
          ];

          buildInputs = [ pkgs.sqlite ];

          buildPhase = ''
            export HOME=$(mktemp -d)

            cp -R ${nodeModules}/node_modules ./
            chmod -R u+w node_modules

            (cd node_modules/better-sqlite3 && node-gyp rebuild --release)
          '';

          installPhase = ''
            mkdir -p $out/lib/qmd
            mkdir -p $out/bin

            cp -r node_modules $out/lib/qmd/
            cp -r src $out/lib/qmd/
            cp package.json $out/lib/qmd/

            makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
              --add-flags "$out/lib/qmd/src/cli/qmd.ts" \
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
            echo "Run: bun src/cli/qmd.ts <command>"
          '';
        };
      }
    );
}
