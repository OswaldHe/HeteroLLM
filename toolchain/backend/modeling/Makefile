# Makefile for roofline modeling tools (Python-based)
# Wrapper for roofline_analyzer.py

PYTHON ?= python3
SCRIPT = roofline_analyzer.py

.PHONY: all test test-innerproduct clean help

all:
	@echo "Roofline analyzer is a Python script, no build required."
	@echo "Run 'make help' for usage."

# Test with inner product profile
test: test-innerproduct

test-innerproduct:
	$(PYTHON) $(SCRIPT) ../compilation/output.cpp \
		--profile-target profile_innerproduct \
		--device fpga_config/u55c.json \
		--resource fpga_config/fpga_resource.json

# Run with JSON output
test-json:
	$(PYTHON) $(SCRIPT) ../compilation/output.cpp \
		--profile-target profile_innerproduct \
		--device fpga_config/u55c.json \
		--resource fpga_config/fpga_resource.json \
		--json

clean:
	@echo "Nothing to clean for Python script."

help:
	@echo "Roofline Modeling Makefile (Python-based)"
	@echo ""
	@echo "This uses profile-based operation counting by running"
	@echo "'make profile_<kernel>' in frontend/dev/profile_test"
	@echo ""
	@echo "Targets:"
	@echo "  test              - Run inner product profile test"
	@echo "  test-innerproduct - Run inner product profile test"
	@echo "  test-json         - Run test with JSON output"
	@echo "  help              - Show this message"
	@echo ""
	@echo "Direct usage:"
	@echo "  $(PYTHON) $(SCRIPT) <source.cpp> --profile-target profile_innerproduct"
	@echo ""
	@echo "Options:"
	@echo "  --profile-target  Make target for profile test (e.g., profile_innerproduct)"
	@echo "  --device          Device config JSON path"
	@echo "  --resource        Resource config JSON path"
	@echo "  --freq            Target frequency in MHz (default: 300)"
	@echo "  --json            Output as JSON only"
	@echo "  --output          Output JSON file path"
