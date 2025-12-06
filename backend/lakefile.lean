import Lake
open Lake DSL

package mathledger where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target] lean_lib ML

-- Keep your sanity exe if you want it
lean_exe ml_sanity where
  root := `Main
