import numpy as np
#Sort following array in to one dimension list
student_scores = np.array([[86, 89], [42,98]])
oned = student_scores.flatten()
print(oned)
#expected output
print(np.sort(oned)[::-1].T)