# Tools for making data frames that can go into KNITR.

result.list <- list()

AppendResult <- function(res) {
  result.list[[length(result.list) + 1]] <<- res
}

ResultRow <- function(analysis, sim.id, variable, measurement, method, component, index1, index2, value) {
  # Produce a dataframe that can be appended to the results.
  AppendResult(data.frame(analysis=analysis,
                          sim.id=sim.id,
                          variable=variable,
                          measurement=measurement,
                          method=method,
                          component=component,
                          index1=index1,
                          index2=index2,
                          value=as.numeric(value)))
}

