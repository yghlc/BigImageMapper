#!/usr/bin/env python
# Filename: basic.py
"""
introduction: support the basic function for program

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 May, 2016
"""

import time,os,sys,subprocess

logfile = 'processLog.txt'
def setlogfile(file_name):
    """
    set log file path
    Args:
        file_name: file path

    Returns: None

    """
    global logfile
    logfile = file_name

def outputlogMessage(message):
    """
    output format log message
    Args:
        message: the message string need to be output

    Returns:None

    """
    global logfile
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
    outstr = timestr +': '+ message
    print(outstr)
    f=open(logfile,'a')
    f.writelines(outstr+'\n')
    f.close()

def stop_and_outputlogMessage(message):
    """
    output format log message and stop program
    :param message:the message string need to be output
    :return:None
    """
    assert False
    outputlogMessage(message)


def output_commandString_from_args_list(args_list):
    commands_str = ''
    if isinstance(args_list,list) and len(args_list)>0:
        for args_str in args_list:
            if ' ' in  args_str:
                commands_str += '\"'+args_str+'\"' + ' ';
            else:
                commands_str += args_str + ' ';
    return commands_str


def exec_command_args_list_one_string(args_list):
    """
    execute a command string
    Args:
        args_list: a list contains args

    Returns: a string

    """
    outputlogMessage(output_commandString_from_args_list(args_list))
    ps = subprocess.Popen(args_list,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # returncode = ps.wait()
    out, err = ps.communicate()
    returncode = ps.returncode
    if returncode is 1:
        outputlogMessage(err.decode())
        return False

    if len(out) > 0:
        return out
    else:
        outputlogMessage('return codes: ' + str(returncode))
        return False

def exec_command_args_list_one_file(args_list,output):
        """
        execute a command string
        Args:
            args_list: a list contains args

        Returns:

        """
        outputlogMessage(output_commandString_from_args_list(args_list))
        ps = subprocess.Popen(args_list)
        returncode = ps.wait()
        if os.path.isfile(output):
            return output
        else:
            outputlogMessage('return codes: '+ str(returncode))
            return False

def exec_command_args_list(args_list):
    """
    execute a command string
    Args:
        args_list: a list contains args

    Returns:

    """
    outputlogMessage(output_commandString_from_args_list(args_list))
    ps = subprocess.Popen(args_list)
    returncode = ps.wait()
    outputlogMessage('return codes: '+ str(returncode))
    return returncode


def exec_command_string(command_str):
    """
    execute a command string
    Args:
        command_str: the command string need to execute

    Returns:(status, result)

    """
    outputlogMessage(command_str)
    (status, result) = getstatusoutput(command_str)
    return status, result

def getstatusoutput(command_str):
    if sys.version_info >= (3, 0):
        (status, result) = subprocess.getstatusoutput(command_str)  # only available in Python 3.x
    else:
        import commands
        (status, result) = commands.getstatusoutput(command_str)

    return (status, result)

def exec_command_string_one_file(command_str,output):
    """
    execute a command string, the result should be a file
    Args:
        command_str:the command string need to execute
        output:the output file path

    Returns:the output file path if successful, False otherwise

    """
    outputlogMessage(command_str)
    # (status, result) = subprocess.check_output(command_str, universal_newlines=True, stderr=sys.stdout)  #available in both Python 2.x and 3.x

    (status, result) = getstatusoutput(command_str)

    if os.path.isfile(output):
        return output
    else:
        outputlogMessage(result)
        # syslog.outputlogMessage('The version of GDAL must be great than 2.0 in order to use the r option ')
        return False

def exec_command_string_output_string(command_str):
    """
    execute a command string, the result should be a string
    Args:
        command_str: the command string need to execute

    Returns:the result string

    """
    outputlogMessage(command_str)
    (status, result) = getstatusoutput(command_str)
    # outputlogMessage(result)
    # if result.find('failed')>=0:
    #     outputlogMessage(result)
    #     return False
    return result


if __name__=='__main__':
    pass