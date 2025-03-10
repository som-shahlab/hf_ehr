VERSION=$(shell grep "version =" pyproject.toml | sed 's/[^0-9.]//g')

release:
	git tag v$(VERSION)
	git push origin main
	git push origin v$(VERSION)
	rm -rf dist/
	python -m build
	twine upload dist/*