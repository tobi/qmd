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
        lib = pkgs.lib;
        isLinux = pkgs.stdenv.isLinux;
        cudaPkgs = if isLinux then pkgs.cudaPackages else null;

        # SQLite with loadable extension support for sqlite-vec
        sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
          configureFlags = (old.configureFlags or []) ++ [
            "--enable-load-extension"
          ];
        });

        mkQmd = {
          nameSuffix ? "",
          extraRuntimeLibs ? [],
          extraWrapperBins ? [],
          extraCmakeIncludeDirs ? [],
          extraCmakeLibraryDirs ? [],
          extraWrapperEnv ? {}
        }:
          pkgs.stdenv.mkDerivation {
            pname = "qmd${nameSuffix}";
            version = "1.0.0";

            src = ./.;

            nativeBuildInputs = [
              pkgs.bun
              pkgs.cmake
              pkgs.makeWrapper
              pkgs.nodejs_22
            ];

            buildInputs = [ pkgs.sqlite ] ++ extraRuntimeLibs;

            buildPhase = ''
              export HOME=$(mktemp -d)
              bun install --frozen-lockfile
            '';

            installPhase = let
              wrapperArgs =
                [
                  "--add-flags \"$out/lib/qmd/src/qmd.ts\""
                  "--prefix PATH : ${lib.makeBinPath ([ pkgs.cmake pkgs.nodejs_22 ] ++ extraWrapperBins)}"
                  "--set DYLD_LIBRARY_PATH \"${lib.makeLibraryPath ([ pkgs.sqlite ] ++ extraRuntimeLibs)}\""
                  "--set LD_LIBRARY_PATH \"${lib.makeLibraryPath ([ pkgs.sqlite ] ++ extraRuntimeLibs)}\""
                ]
                ++ lib.optional (extraCmakeIncludeDirs != [])
                  "--prefix CMAKE_INCLUDE_PATH : ${lib.concatStringsSep ":" extraCmakeIncludeDirs}"
                ++ lib.optional (extraCmakeLibraryDirs != [])
                  "--prefix CMAKE_LIBRARY_PATH : ${lib.concatStringsSep ":" extraCmakeLibraryDirs}"
                ++ (lib.mapAttrsToList (k: v: "--set ${k} \"${v}\"") extraWrapperEnv);
            in ''
              mkdir -p $out/lib/qmd
              mkdir -p $out/bin

              cp -r node_modules $out/lib/qmd/
              cp -r src $out/lib/qmd/
              cp package.json $out/lib/qmd/

              makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
                ${lib.concatStringsSep " \\\n  " wrapperArgs}
            '';

            meta = with pkgs.lib; {
              description = "On-device search engine for markdown notes, meeting transcripts, and knowledge bases";
              homepage = "https://github.com/tobi/qmd";
              license = licenses.mit;
              platforms = platforms.unix;
            };
          };

        qmd = mkQmd { };
        qmdVulkan = mkQmd {
          nameSuffix = "-vulkan";
          extraRuntimeLibs = lib.optionals isLinux [
            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.shaderc
          ];
          extraWrapperBins = lib.optionals isLinux [ pkgs.shaderc ];
          extraCmakeIncludeDirs = lib.optionals isLinux [ "${pkgs.vulkan-headers}/include" ];
          extraCmakeLibraryDirs = lib.optionals isLinux [ "${pkgs.vulkan-loader}/lib" ];
          extraWrapperEnv = lib.optionalAttrs isLinux {
            NODE_LLAMA_CPP_CMAKE_OPTION_CMAKE_CXX_FLAGS = "-include cstdint";
            CXXFLAGS = "-include cstdint";
          };
        };
        qmdCuda = mkQmd {
          nameSuffix = "-cuda";
          extraRuntimeLibs = lib.optionals isLinux [ cudaPkgs.cudatoolkit ];
          extraWrapperBins = lib.optionals isLinux [ cudaPkgs.cudatoolkit ];
        };

        baseShellInputs = [
          pkgs.bun
          pkgs.cmake
          pkgs.makeWrapper
          pkgs.nodejs_22
          sqliteWithExtensions
        ];

        mkShell = {
          name,
          extraInputs ? [],
          extraRuntimeLibs ? [],
          extraCmakeIncludeDirs ? [],
          extraCmakeLibraryDirs ? [],
          extraEnv ? {},
          extraShellHook ? ""
        }:
          pkgs.mkShell {
            buildInputs = baseShellInputs ++ extraInputs ++ extraRuntimeLibs;
            shellHook = ''
              export BREW_PREFIX="''${BREW_PREFIX:-${sqliteWithExtensions.out}}"
              export LD_LIBRARY_PATH="${lib.makeLibraryPath ([ sqliteWithExtensions ] ++ extraRuntimeLibs)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export DYLD_LIBRARY_PATH="${lib.makeLibraryPath ([ sqliteWithExtensions ] ++ extraRuntimeLibs)}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
              ${lib.optionalString (extraCmakeIncludeDirs != []) ''
                export CMAKE_INCLUDE_PATH="${lib.concatStringsSep ":" extraCmakeIncludeDirs}''${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
              ''}
              ${lib.optionalString (extraCmakeLibraryDirs != []) ''
                export CMAKE_LIBRARY_PATH="${lib.concatStringsSep ":" extraCmakeLibraryDirs}''${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
              ''}
              ${lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "export ${k}=\"${v}\"") extraEnv)}
              ${extraShellHook}
              echo "QMD development shell (${name})"
              echo "Run: bun src/qmd.ts <command>"
            '';
          };
      in
      {
        packages = {
          default = qmd;
          qmd = qmd;
        }
        // lib.optionalAttrs isLinux { "qmd-vulkan" = qmdVulkan; }
        // lib.optionalAttrs isLinux { "qmd-cuda" = qmdCuda; };

        apps.default = {
          type = "app";
          program = "${qmd}/bin/qmd";
        };

        devShells = {
          default = mkShell { name = "cpu"; };
          cpu = mkShell { name = "cpu"; };
        }
        // lib.optionalAttrs isLinux {
          vulkan = mkShell {
            name = "vulkan";
            extraRuntimeLibs = [
              pkgs.vulkan-loader
              pkgs.vulkan-headers
              pkgs.shaderc
            ];
            extraInputs = [ pkgs.shaderc ];
            extraCmakeIncludeDirs = [ "${pkgs.vulkan-headers}/include" ];
            extraCmakeLibraryDirs = [ "${pkgs.vulkan-loader}/lib" ];
            extraEnv = {
              NODE_LLAMA_CPP_CMAKE_OPTION_CMAKE_CXX_FLAGS = "-include cstdint";
              CXXFLAGS = "-include cstdint";
            };
            extraShellHook = ''
              if [ -d /run/opengl-driver/share/vulkan/icd.d ]; then
                icd_files=$(ls /run/opengl-driver/share/vulkan/icd.d/*.json 2>/dev/null | paste -sd ":" -)
                if [ -n "$icd_files" ]; then
                  export VK_ICD_FILENAMES="$icd_files"
                fi
              fi
            '';
          };
        }
        // lib.optionalAttrs isLinux {
          cuda = mkShell {
            name = "cuda";
            extraRuntimeLibs = [ cudaPkgs.cudatoolkit ];
            extraInputs = [ cudaPkgs.cudatoolkit ];
            extraShellHook = ''
              export CUDA_PATH="${cudaPkgs.cudatoolkit}"
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi
            '';
          };
        };
      }
    );
}
