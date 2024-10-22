import platform

def adjustFilepathsForOS(listOfFilepaths):
  
  returnListOfFilepaths = []
  osType = getOS()  

  for filepath in listOfFilepaths:
    
    tempString = ''
    filepathLength =  len(filepath)

    for i in range(filepathLength):
      
      if filepath[i] == '\\' or filepath[i] == '/':
      
        if osType == 0:
        
          tempString += '\\'

        else:
          
          tempString += '/'

      else:

        tempString += filepath[i]
    
    returnListOfFilepaths.append(tempString)

  return returnListOfFilepaths

def getOS():

  osName = platform.system()

  match osName:

    case 'Windows':

      return 0

    case 'Darwin':

      return 1

    case 'Linux':

      return 2

    case _:

      return -1


def main():

  test = ['hello\\my\\name\\is\\john', '\\']
  print(adjust_filepaths_for_OS(test))

if __name__ == "__main__":
  main()

