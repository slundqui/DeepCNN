function writeMat(inMat, filename)
   data = single(inMat);
   save(filename, 'data', '-v6');
end
