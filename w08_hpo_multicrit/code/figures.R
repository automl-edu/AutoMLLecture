library(ggplot2)

fun = function(x) (x - 1)^2
p = ggplot(data.frame(x = c(0, 3)), aes(x)) + stat_function(fun = fun)
p = p + geom_point(x = 1, y = 0, color = "green", size = 3)
p = p + theme_bw() + ylab("c") + xlab(expression(lambda))
ggsave(p, filename = "../images/graph1.png", height = 2, width = 2)

fun1 = function(x) (x - 1)^2
fun2 = function(x) 3 * (x - 2)^2
p = ggplot(data.frame(x = c(0, 3)), aes(x)) + stat_function(fun = fun1) + stat_function(fun = fun2, color = "blue")
p = p + theme_bw() +  ylab("c") + xlab(expression(lambda))

ggsave(p, filename = "../images/graph2.png", height = 2, width = 2)

x = seq(0, 3, length.out = 1000)
xpareto = seq(1, 2, length.out = 1000)

p2 = ggplot() + geom_point(data = data.frame(c1 = fun1(x), c2 = fun2(x)), aes(x = c1, y = c2), size = 0.05) + geom_point(data = data.frame(c1 = fun1(xpareto), c2 = fun2(xpareto)), aes(x = c1, y = c2), color = "green", size = 0.05) + theme_bw()

ggsave(p2, filename = "../images/graph3.png", height = 2, width = 2)
