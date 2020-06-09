library(ggplot2)
library(gridExtra)

df = readRDS("expedia_example.rds")

p = ggplot(data = df, aes(x = mean_price, y = - mean_rating)) + geom_point(size = 1.5)
p = p + theme_bw()
p = p + ylim(c(- 5.5, -2))
p = p + xlab("Price per night") + ylab("Rating")
p

p1 = ggplot(data = df, aes(x = mean_price, y = - mean_rating)) + geom_point(size = 1.5)
p1 = p1 + geom_point(data = df[16:17, ], aes(x = mean_price, y = - mean_rating), size = 2, colour = c("green", "red"))
p1 = p1 + theme_bw()
p1 = p1 + ylim(c(- 5.5, -2))
p1 = p1 + xlab("Price per night") + ylab("Rating")

p2 = ggplot(data = df, aes(x = mean_price, y = - mean_rating)) + geom_point(size = 2)
p2 = p2 + geom_point(data = df[c(10, 16), ], aes(x = mean_price, y = - mean_rating), size = 2, colour = "orange")
p2 = p2 + theme_bw()
p2 = p2 + ylim(c(-5.5, -2))
p2 = p2 + xlab("Price per night") + ylab("Rating")

grid.arrange(p1, p2, ncol = 2)

p1

p2

df$mean_rating = - df$mean_rating
P = df[order(df$mean_rating, df$mean_price,decreasing=FALSE),]
P = P[which(!duplicated(cummin(P$mean_price))),]

p2 = ggplot(data = df, aes(x = mean_price, y = mean_rating)) + geom_point(size = 2)
p2 = p2 + geom_point(data = P, aes(x = mean_price, y = mean_rating), size = 2, colour = "orange")
p2 = p2 + geom_line(data = P, aes(x = mean_price, y = mean_rating), colour = "orange")
p2 = p2 + theme_bw()
p2 = p2 + ylim(c(-5, -2))
p2 = p2 + xlab("Price per night") + ylab("Rating")

p2

fun = function(x) (x - 1)^2
p = ggplot(data.frame(x = c(0, 3)), aes(x)) + stat_function(fun = fun)
p = p + geom_point(x = 1, y = 0, color = "green", size = 3)
p = p + theme_bw() + ylab("c") + xlab(expression(lambda))
p

fun1 = function(x) (x - 1)^2
fun2 = function(x) 3 * (x - 2)^2
p = ggplot(data.frame(x = c(0, 3)), aes(x)) + stat_function(fun = fun1) + stat_function(fun = fun2, color = "blue")
p = p + theme_bw()
p

x = seq(0, 3, length.out = 1000)
xpareto = seq(1, 2, length.out = 1000)

p2 = ggplot() + geom_point(data = data.frame(f1 = fun1(x), f2 = fun2(x)), aes(x = f1, y = f2), size = 0.05) + geom_point(data = data.frame(f1 = fun1(xpareto), f2 = fun2(xpareto)), aes(x = f1, y = f2), color = "green", size = 0.05) + theme_bw()
p2

df$apriori = df$mean_price + 50 * df$mean_rating

p1 = ggplot()
p1 = p1 + geom_point(data = df, aes(x = apriori, y = 0), size = 2)
p1 = p1 + geom_point(data = df[which.min(df$apriori), ], aes(x = apriori, y = 0), colour = "green", size = 2)
p1 = p1 + theme_bw()
p1 = p1 + xlab("Weighted sum")
p1 = p1 + theme(axis.title.y = element_blank(),
                axis.text.y = element_blank(),
                axis.ticks.y = element_blank())

p2 = ggplot(data = df, aes(x = mean_price, y = mean_rating)) + geom_point(size = 2)
p2 = p2 + geom_point(data = df[which.min(df$apriori), ], aes(x = mean_price, y = mean_rating), size = 2, colour = "green")
p2 = p2 + theme_bw()
p2 = p2 + ylim(c(-5, -2))
p2 = p2 + xlab("Price per night") + ylab("Rating")

grid.arrange(p1, p2, ncol = 2)

p1 = ggplot(data = df, aes(x = mean_price, y = mean_rating)) + geom_point(size = 2)
p1 = p1 + geom_point(data = df[df$mean_rating == -5, ], aes(x = mean_price, y = mean_rating), size = 2, colour = "orange")
p1 = p1 + theme_bw()
p1 = p1 + ylim(c(-5, -2))
p1 = p1 + ggtitle("1) max. rating")
p1 = p1 + xlab("Price per night") + ylab("Rating")

p2 = p1 + geom_point(data = df[(df$mean_rating == - 5.0 & df$mean_price < 150), ], colour = "green", size = 2)
p2 = p2 + ggtitle("2) min. price")

grid.arrange(p1, p2, ncol = 2)

P = df[order(df$mean_rating, df$mean_price,decreasing=FALSE),]
P = P[which(!duplicated(cummin(P$mean_price))),]

p1 = ggplot(data = df, aes(x = mean_price, y = mean_rating)) + geom_point(size = 2)
p1 = p1 + geom_point(data = P, aes(x = mean_price, y = mean_rating), size = 2, colour = "orange")
p1 = p1 + geom_line(data = P, aes(x = mean_price, y = mean_rating), colour = "orange")
p1 = p1 + theme_bw()
p1 = p1 + ylim(c(-5, -2))
p1 = p1 + xlab("Price per night") + ylab("Rating")


p2 = p1 + geom_point(data = P[P$mean_rating == -4.5, ], aes(x = mean_price, y = mean_rating), colour = "green", size = 2)

g = grid.arrange(p1, p2, ncol = 2)
ggsave("../images/expedia-11-1.pdf", g, height = 2, width = 5)

x = seq(-1, 4, length.out = 1000)
lin = 3 * 0.4 - 2 * x

p2 = ggplot() + geom_point(data = data.frame(f1 = fun1(x), f2 = fun2(x)), aes(x = f1, y = f2), size = 0.7)
p2 = p2 + geom_line(aes(x = x, y = lin)) + ylim(c(-3, 25))
p2 = p2 + geom_point(aes(x = 0.36, y = 0.48), colour = "green", size = 3)
p2 = p2 + theme_bw()
p2

f1 = function(x) 0.01 * sum(x^2) - 2
f2 = function(x) 0.01 * sum(c(0.1, 0.3) * (x - c(-10, 20))^2)

x1 = x2 = seq(-10, 20, length.out = 100)
grid = expand.grid(x1 = x1, x2 = x2)
grid$y1 = apply(grid[, 1:2], 1, f1)
grid$y2 = apply(grid[, 1:2], 1, f2)

melt = reshape2::melt(grid, id.vars = c("x1", "x2"))

p = ggplot(data = melt) + geom_raster(aes(x = x1, y = x2, fill = value))
p = p + geom_contour(aes(x = x1, y = x2, z = value, colour = variable), bins = 15)
p = p + ylim(c(-20, 40)) + xlim(c(-20, 40)) + theme_bw()
p
