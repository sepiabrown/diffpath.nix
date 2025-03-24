{
  description = "diffpath flake using uv2nix";

  inputs = {
    nixpkgs.url = "github:sepiabrown/nixpkgs/cusparselt-init";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      # Load a uv workspace from a workspace root.
      # Uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Create package overlay from workspace.
      overlay = workspace.mkPyprojectOverlay {
        # Prefer prebuilt binary wheels as a package source.
        # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
        # Binary wheels are more likely to, but may still require overrides for library dependencies.
        sourcePreference = "wheel"; # or sourcePreference = "sdist";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups
      #
      # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
      # This is an additional overlay implementing build fixups.
      # See:
      # - https://pyproject-nix.github.io/uv2nix/FAQ.html
      cudaLibs = [
        (lib.getOutput "stubs" pkgs.cudaPackages_11_7.cuda_cudart)
      ] ++ map lib.getLib [
        pkgs.cudaPackages_11_7.nccl
        pkgs.cudaPackages_11_7.cudatoolkit
        pkgs.cudaPackages_11_7.cuda_cupti
        pkgs.cudaPackages_11_7.cudnn
      ];

      pyprojectOverrides = final: prev: {
        torch = prev.torch.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
        });
        torchvision = prev.torchvision.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;# ++ [ final.torch ];

          postFixup = ''
            addAutoPatchelfSearchPath "${final.torch}"
          '';
        });
        nvidia-cusolver-cu11 = prev.nvidia-cusolver-cu11.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
        });
        nvidia-cudnn-cu11 = prev.nvidia-cudnn-cu11.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ cudaLibs;
        });
        mpi4py = prev.mpi4py.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            (lib.getDev pkgs.mpich)
          ];
          nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [
            pkgs.pytestCheckHook
            pkgs.mpiCheckPhaseHook
          ];
        });
      };

      buildSystemOverrides = final: prev:
      let
        deps = {
          lit = {
            setuptools = [ ];
            wheel = [ ];
          };
          mpi4py = {
            setuptools = [ ];
            wheel = [ ];
          };
        };
      in
      lib.mapAttrs (
        name: spec:
        prev.${name}.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem spec;
        })
      ) deps;

      # This example is only using x86_64-linux
      # pkgs = nixpkgs.legacyPackages.x86_64-linux;
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };

      # Use Python 3.9 from nixpkgs
      python = pkgs.python39;

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
              buildSystemOverrides
            ]
          );

    in
    {
      inherit pkgs;
      # Package a virtual environment as our main application.
      #
      # Enable no optional dependencies for production build.
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "diffpath-env" (workspace.deps.default // {
        # torch = [ ];
      });

      # Make hello runnable with `nix run`
      apps.x86_64-linux =
        let
          REMOTE_HOST="2.0.0.12";
          REMOTE_PORT="18888";
          LOCAL_PORT="18888";
        in
      {
        default = self.apps.x86_64-linux.single-gpu;
        train = {
          type = "app";
          program = "${self.packages.x86_64-linux.default}/bin/train_script";
        };
        single-gpu = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "single-gpu";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              export NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
              exec train_script \
                --config configs/train_config.yaml \
                --data_dir data \
                --dataset svhn
            '';
            inheritPath = true;
          }}/bin/single-gpu";
        };
        multi-gpus = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "multi-gpus";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              export NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
              exec torchrun \
                --standalone \
                --nproc_per_node=2 \
                ${self.packages.x86_64-linux.default}/bin/train_script \
                --config configs/train_config.yaml \
                --data_dir data \
                --dataset celeba
            '';
            inheritPath = true;
          }}/bin/multi-gpus";
        };
        save-train-statistics = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "save-train-statistics";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              export NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
              exec save_train_statistics_script \
                --model celeba \
                --data_dir data \
                --model_path results/celeba/2025_03_21_025840/model099999.pt \
                --config configs/celeba_model_config.yaml \
                --batch_size 256 \
                --n_ddim_steps 10 \
                --device 0 \
                --dataset "$@"
            '';
            inheritPath = true;
          }}/bin/save-train-statistics";
        };
        save-test-statistics = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "save-test-statistics";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              export NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
              exec save_test_statistics_script \
                --model celeba \
                --data_dir data \
                --model_path results/celeba/2025_03_21_025840/model099999.pt \
                --config configs/celeba_model_config.yaml \
                --batch_size 256 \
                --n_ddim_steps 10 \
                --device 0 \
                --dataset "$@"
            '';
            inheritPath = true;
          }}/bin/save-test-statistics";
        };
        jupyter = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "start-jupyter";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              exec jupyter lab --port=${REMOTE_PORT} "$@"
            '';
            inheritPath = true;
          }}/bin/start-jupyter";
        };
        port-forward = {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "port-forward";
            runtimeInputs = [ self.packages.x86_64-linux.default ];
            text = ''
              ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_HOST}
            '';
            inheritPath = true;
          }}/bin/port-forward";
        };
      };

      # This example provides two different modes of development:
      # - Impurely using uv to manage virtual environments
      # - Pure development using uv2nix to manage virtual environments
      devShells.x86_64-linux = {
        # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
        # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
        impure = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
          ];
          env =
            {
              # Prevent uv from managing Python downloads
              UV_PYTHON_DOWNLOADS = "never";
              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath (__filter (p: p.pname != "glibc") pkgs.pythonManylinuxPackages.manylinux1);
            };
          shellHook = ''
            unset PYTHONPATH
          '';
        };

        # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
        # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
        #
        # This means that any changes done to your local files do not require a rebuild.
        #
        # Note: Editable package support is still unstable and subject to change.
        uv2nix =
          let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "hello-world" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope (
              lib.composeManyExtensions [
                editableOverlay

                # Apply fixups for building an editable package of your workspace packages
                (final: prev: {
                  diffpath = prev.diffpath.overrideAttrs (old: {
                    # It's a good idea to filter the sources going into an editable build
                    # so the editable package doesn't have to be rebuilt on every change.
                    src = lib.fileset.toSource {
                      root = old.src;
                      fileset = lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                        (old.src + "/improved_diffusion/__init__.py")
                      ];
                    };

                    # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                    #
                    # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                    # This behaviour is documented in PEP-660.
                    #
                    # With Nix the dependency needs to be explicitly declared.
                    nativeBuildInputs =
                      old.nativeBuildInputs
                      ++ final.resolveBuildSystem {
                        editables = [ ];
                      };
                  });

                })
              ]
            );

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "diffpath-dev-env" workspace.deps.all;

          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              # Don't create venv using uv
              UV_NO_SYNC = "1";

              # Force uv to use Python interpreter from venv
              UV_PYTHON = "${virtualenv}/bin/python";

              # Prevent uv from downloading managed Python's
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH

              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel) NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
            '';
          };
      };
    };
}
