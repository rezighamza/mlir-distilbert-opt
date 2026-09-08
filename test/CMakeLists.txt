set(DISTILBERT_TEST_DEPENDS
  distilbert-opt
  distilbert-translate
  FileCheck
  count
  not
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

add_lit_testsuite(check-distilbert "Running the DistilBERT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DISTILBERT_TEST_DEPENDS}
)
set_target_properties(check-distilbert PROPERTIES FOLDER "Tests")
