library(ggplot2)
# 100 epochs (5 by 5)
test0.01700 = c(11.3, 7.86, 6.6, 5.46, 4.77, 4.31, 3.81, 3.5, 3.28, 3.05, 2.88, 2.75, 2.67, 2.56, 2.43, 2.42, 2.38, 2.31, 2.28, 2.26, 2.24)
test0.1700 = c(8.25,3.15,2.39,2.11,2.02,1.9,1.76,1.71,1.74,1.77,1.72,1.72,1.71,1.71,1.7,1.69,1.66,1.63,1.63,1.65,1.63)
test0.1256 = c(7.57,3.24,2.39,2.21,2.16,2.02,1.96,1.92,1.96,1.97,1.96,1.91,1.89,1.85,1.86,1.83,1.81,1.81,1.83,1.83,1.82)
test0.05256eta0.05 = c(7.62,3.21,2.45,2.24,2.17,2.07,1.97,1.98,1.95,1.92,1.91,1.83,1.89,1.89,1.86,1.81,1.81,1.82,1.84,1.85,1.83)
test0.1256eta0.1 = c(6.22,2.51,2.24,2.01,1.92,1.81,1.75,1.74,1.75,1.8,1.8,1.75,1.74,1.73,1.74,1.7,1.73,1.75,1.73,1.74,1.72)
test0.1700eta0.1 = c(7.09,2.52,2.07,1.85,1.87,1.77,1.71,1.67,1.64,1.64,1.62,1.63,1.63,1.6,1.61,1.62,1.63,1.61,1.62,1.64,1.61)
X = seq(0,100,5)

# Plot learning rate differences
P <- ggplot(NULL, aes(X))
P <- P + geom_point(aes(y = test0.1700, color="0.1")) + geom_line(aes(y = test0.1700, color="0.1"))
P <- P + geom_point(aes(y = test0.01700, color="0.01")) + geom_line(aes(y = test0.01700, color="0.01"))
P <- P + labs(color = 'tasa') + ylab('tasa de error en test (%)') + xlab('pasadas al conjunto')
P

#Plot neurons amount difference
amount <- ggplot(NULL, aes(X))
amount <- amount + geom_point(aes(y = test0.1700, color="700")) + geom_line(aes(y = test0.1700, color="700"))
amount <- amount + geom_point(aes(y = test0.1256, color="256")) + geom_line(aes(y = test0.1256, color="256"))
amount <- amount + labs(color = 'nº neuronas') + ylab('tasa de error en test (%)') + xlab('pasadas al conjunto')

#Momentums comparation
momentums <- ggplot(NULL, aes(X))
momentums <- momentums + geom_point(aes(y = test0.05256eta0.05, color="n = 256, lr = 0.05, mr = 0.05")) + geom_line(aes(y = test0.05256eta0.05, color="n = 256, lr = 0.05, mr = 0.05"))
momentums <- momentums + geom_point(aes(y = test0.1256eta0.1, color="n = 256, lr = 0.1, mr = 0.1")) + geom_line(aes(y = test0.1256eta0.1, color="n = 256, lr = 0.1, mr = 0.1"))
momentums <- momentums + geom_point(aes(y = test0.1700eta0.1, color="n = 700, lr = 0.1, mr = 0.1")) + geom_line(aes(y = test0.1700eta0.1, color="n = 700, lr = 0.1, mr = 0.1"))
momentums <- momentums + labs(color = 'configuración') + ylab('tasa de error en test (%)') + xlab('pasadas al conjunto')

#Rates comparation
rates = c(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1)
errorRates = c(6.3, 4.8, 3.26, 2.53, 2.31, 2.19, 2.29, 2.33, 3.07)

p <- ggplot(NULL, aes(rates))
p <- p + geom_point(aes(y = errorRates, colour = "blue")) + geom_line(aes(y = errorRates, colour = "blue")) + ylab('tasa de error en test (%)') + xlab('tasa de aprendizaje')
p <- p + scale_color_manual(values=c("#CC6666"), guide = FALSE, labels = rates)