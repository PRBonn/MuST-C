""" function to convert DNs to radiance
"""

def convert_DNs_to_radiance():
    doc=Metashape.app.document 
    chunk=doc.chunk sensors=chunk.sensors 
    for i in range (0,10): 
        sensors[i].normalize_sensitivity=True 
        sensors[i].normalize_to_float=True
