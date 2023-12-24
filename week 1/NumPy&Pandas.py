
## Question: Reverse a string using a for loop in Python.


def reverse_str(input_str):
    s = ''
    for char in input_str:
        s = char + s
    return s

input_str = input('Enter a string : ')
print("Reversed string :",reverse_str(input_str))

## Question: Write a Python program to find the sum of all numbers in a list using a for loop.

user_input = input("Enter a list of integers :")
num_list = [int(num) for num in user_input.split(' ')]
sum = 0
for e in num_list:
    sum += e
print("Sum :",sum)

## Question: Write a Python program that checks whether a given number is even or odd using an if-else statement.

num= int(input("Enter a number :"))
if num%2 == 0:
    print(num, 'is Even Number')
else:
    print(num,"is Odd Number")

## Question: Implement a program to determine if a year is a leap year or not using if-elif-else statements.

def leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return "Leap Year"
    else:
        return "Not a Leap Year"

year = int(input('Enter the year :'))
print(year,' is ',leap_year(year))

## Question: Use a lambda function to square each element in a list

user_input = input("Enter a list of integers :")
num_list = [int(num) for num in user_input.split(' ')]
squared_list = list(map(lambda x: x**2, num_list))
print(squared_list)

## Question: Write a lambda function to calculate the product of two

x = int(input("Enter the first number: "))
y = int(input("Enter the second number: "))

pdt = lambda x, y: x * y

print('Product :',pdt(x,y))