model.js     the original Cacatoo model, untouched
cacatoo.js   the Cacatoo library it needs
model_ref.js generated from model.js by ../test_scripts/make_model_ref.py:
             identical, but with every parameter exposed on the command line so
             the Python port can be compared against it across the whole
             parameter space. Regenerate it rather than editing it.

To run these you need node and yargs 17:   npm install yargs@17
(yargs 18+ is ESM-only; model.js uses require().)

Note also that model.js will not create its output directory -- it appends with
fs.appendFileSync, which fails if the directory does not already exist.
