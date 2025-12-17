import Lake
open Lake DSL

package mathledger where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a3e910d1569d6b943debabe63afe6e3a3d4061ff"

@[default_target] lean_lib ML

lean_exe ml_sanity where
  root := `Main
