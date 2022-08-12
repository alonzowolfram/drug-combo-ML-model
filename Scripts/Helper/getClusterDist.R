# https://www.biostars.org/p/13143/#13172
getClusterDist <- function(dat, clusters) {
  # 1) Find the centroid of each cluster. 
  ind_1 <- (clusters == 1)
  ind_2 <- (clusters == 2)
  
  centroid_1 <- colMeans(dat[ind_1,])
  centroid_2 <- colMeans(dat[ind_2,])
  
  # 2) Bind the two vectors (each of which corresponds to a centroid) into a matrix so we can calculate the distance. 
  m <- rbind(centroid_1, centroid_2)
  
  # 3) Calculate the distance between the centroids. This is the value to be returned.
  return(dist(m))
}