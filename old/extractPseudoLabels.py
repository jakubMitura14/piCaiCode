#below extracting the pseudo labels from I suppose class activation maps
#https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/6909800fbc17bc3d833d91e4977d3baf47975fda/src/report_guided_annotation/create_automatic_annotations.py
# PI-RADS >= 3 lesions are retained

# important beware the shape is strange ! - like reading directly simple itk to numpy
# probably will be simpler to write softmaxes to folder and then read it from there for loss ?