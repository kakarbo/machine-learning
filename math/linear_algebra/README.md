# Algebra Lineal

En este proyecto aprenderemos que es un vecto, que es una matriz, que es una transposición
que forma tiene una matriz, que es un eje, que es un slice y para que sirve, como se corta un vector/matriz, cuales son las operaciones con elementos, como se puede concatenar vectores y matrices, que es un producto punto, como se multiplica una matriz, y utilizaremos numpy para realizar las operaciones mencionandas anteriormente, además aprenderemos como se paraleliza y por que es importante en machine learning, también veremos que es un broadcasting

## Recursos
#### Leer o mirar
* [Introduction to vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
* [What is a matrix?](https://math.stackexchange.com/questions/2782717/what-exactly-is-a-matrix) (not [the matrix](https://www.imdb.com/title/tt0133093/))
* [Transpose](https://en.wikipedia.org/wiki/Transpose)
* [Understanding the dot product](https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)
* [Matrix Multiplication](https://www.youtube.com/watch?v=BzWahqwaS8k)
* [What is the relationship between matrix multiplication and the dot product?](https://www.quora.com/What-is-the-relationship-between-matrix-multiplication-and-the-dot-product)
* [The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices]((https://www.youtube.com/watch?v=rW2ypKLLxGk)) (advanced)
* [numpy tutorial](https://numpy.org/doc/stable/user/quickstart.html) (until Shape Manipulation (excluded))
* [numpy basics](https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html) (until Universal Functions (included))
* [array indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing)
* [numerical operations on arrays](http://scipy-lectures.org/intro/numpy/operations.html)
* [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
* [numpy mutations and broadcasting](https://towardsdatascience.com/two-cool-features-of-python-numpy-mutating-by-slicing-and-broadcasting-3b0b86e8b4c7)

### Referencias
* [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
* [numpy.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
* [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
* [numpy.ndarray.transpose](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html)
* [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

## Objetivos de aprendizaje
### General
* What is a vector?
* What is a matrix?
* What is a transpose?
* What is the shape of a matrix?
* What is an axis?
* What is a slice?
* How do you slice a vector/matrix?
* What are element-wise operations?
* How do you concatenate vectors/matrices?
* What is the dot product?
* What is matrix multiplication?
* What is Numpy?
* What is parallelization and why is it important?
* What is broadcasting?

## Requerimientos
### Python Scripts
* Editores permitidos: ```vi```, ```vim```, ```emacs```
* Todos tus archivos serán interpretados/compilados en Ubuntu 20.04 LTS usando ```python3``` (versión 3.8)
* Tus archivos se ejecutarán con ```numpy``` (versión 1.19.2)
* Todos tus archivos deben terminar con una nueva línea
* La primera línea de todos tus archivos debe ser exactamente ```#!/usr/bin/env python3```
* Un archivo ```README.md```, en la raíz de la carpeta del proyecto, es obligatorio
* Tu código debe seguir ```pycodestyle``` (versión 2.6)
* Todos tus módulos deben tener documentación (```python3 -c 'print(__import__("mi_módulo").__doc__)'```)
* Todas tus clases deben tener documentación (```python3 -c 'print(__import__("mi_modulo").MiClase.__doc__)'```)
* Todas tus funciones (dentro y fuera de una clase) deben tener documentación (```python3 -c 'print(__import__("mi_modulo").mi_funcion.__doc__)'``` y ```python3 -c 'print(__import__("mi_modulo").MiClase.mi_funcion.__doc__)'```)
* **A menos que se indique lo contrario, no está permitido importar ningún módulo**
* Todos los archivos deben ser ejecutables
* La longitud de los archivos se comprobará con ```wc```

## Más info
### Instalación ubuntu 20.04 y python3.8
Siga las instrucciones listadas en Vagrant Usando en su ordenador personal, debería estar usando ubuntu/focal64.

Python 3.8 viene preinstalado en Ubuntu 20.04. ¡Qué conveniente! Puedes confirmarlo con python3 -V

### Instalación pip (latest)
[pip installation](https://pip.pypa.io/en/stable/installation/)

### Instalación numpy 1.19.2, scipy 1.6.2 y pycodestyle 2.6
```
$ pip install --user numpy==1.19.2
$ pip install --user scipy==1.6.2
$ pip install --user pycodestyle==2.6
```
Para comprobar que todos se han descargado correctamente, utilice la lista pip.

## Tareas
### 0. Slice Me up
Complete the following source code (found below):

* arr1 should be the first two numbers of arr
* arr2 should be the last five numbers of arr
* arr3 should be the 2nd through 6th numbers of arr
* You are not allowed to use any loops or conditional statements
* Your program should be exactly 8 lines
```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 0-slice_me_up.py 
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
alexa@ubuntu-focal:0x00-linear_algebra$ ./0-slice_me_up.py 
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
alexa@ubuntu-focal:0x00-linear_algebra$ wc -l 0-slice_me_up.py 
8 0-slice_me_up.py
alexa@ubuntu-focal:0x00-linear_algebra$ 
```

### 1. Trim Me Down
Complete the following source code (found below):

* the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
* You are not allowed to use any conditional statements
* You are only allowed to use one for loop
* Your program should be exactly 6 lines
```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 1-trim_me_down.py 
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
# your code here
print("The middle columns of the matrix are: {}".format(the_middle))
alexa@ubuntu-focal:0x00-linear_algebra$ ./1-trim_me_down.py 
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
alexa@ubuntu-focal:0x00-linear_algebra$ wc -l 1-trim_me_down.py 
6 1-trim_me_down.py
alexa@ubuntu-focal:0x00-linear_algebra$
```

### 2. Size Me Please
Write a function def matrix_shape(matrix): that calculates the shape of a matrix:

* You can assume all elements in the same dimension are of the same type/shape
* The shape should be returned as a list of integers
```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))
alexa@ubuntu-focal:0x00-linear_algebra$ ./2-main.py 
[2, 2]
[2, 3, 5]
alexa@ubuntu-focal:0x00-linear_algebra$ 
```
