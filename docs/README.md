## Compiling the docs
- Make sure you are in `docssrc`, then follow the instructions under `run` in our [documentation building github actions job](https://github.com/mindsdb/sigmoid/blob/staging/.github/workflows/doc_build.yml)
- Then go into the newly built docs and start a server to see them: `cd ../docs && python -m http.server`
- Docs should now be available at: 0.0.0.0:8000 | Alternatively, you can just open the `index.html` with a browser and that should work too

## Ref

For how autosummary works: https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough

## Manual steps

Currently notebooks have to be built manually using: `find . -iname '*.ipynb' -exec jupyter nbconvert --to notebook --inplace --execute {} \;`