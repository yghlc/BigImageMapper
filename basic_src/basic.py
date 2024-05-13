#!/usr/bin/env python
# Filename: basic.py
"""
introduction: support the basic function for program

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 May, 2016
"""

import time,os,sys,subprocess
import psutil

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
    if returncode != 0:
        outputlogMessage(err.decode())
        return False

    if len(out) > 0:
        return out
    else:
        outputlogMessage('return codes: ' + str(returncode))
        return False

def exec_command_args_list_one_file(args_list,output,b_verbose=True):
        """
        execute a command string
        Args:
            args_list: a list contains args

        Returns:

        """
        if b_verbose:
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

def os_system_exit_code(command_str):
    '''
    run a common string, check the exit code
    :param command_str:
    :return:
    '''
    res = os.system(command_str)
    if res != 0:
        print('command_str:')
        print(command_str)
        sys.exit(1)

def exec_command_string(command_str):
    """
    execute a command string
    Args:
        command_str: the command string need to execute

    Returns:(status, result)

    """
    print(command_str)
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
    print(command_str)
    # (status, result) = subprocess.check_output(command_str, universal_newlines=True, stderr=sys.stdout)  #available in both Python 2.x and 3.x

    (status, result) = getstatusoutput(command_str)
    if status != 0:
        outputlogMessage(result)

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
    print(command_str)
    (status, result) = getstatusoutput(command_str)
    # outputlogMessage(result)
    # if result.find('failed')>=0:
    #     outputlogMessage(result)
    #     return False
    return result

def b_all_process_finish(processes):
    for task in processes:
        if task.is_alive():
            return False
    return True

def alive_process_count(processes):
    count = 0
    for task in processes:
        if task.is_alive():
            count += 1
    return count


def close_remove_completed_process(processes):
    # close process to release resource, avoid error of "open too many files"
    # will modify the list: processes
    for task in processes:
        if task.is_alive() is False:
            task.close()
            processes.remove(task)

def check_exitcode_of_process(processes):
    # check exitcode, if not 0, the quit
    for task in processes:
        if task.exitcode is not None and task.exitcode != 0:
            print('a process was failed, exitcode:',task.exitcode,'process id:',task.pid)
            sys.exit(task.exitcode)


def get_curr_process_openfiles():
    # the the open files by current process
    # if want to check all the open files in a system, need go through psutil.process_iter()
    proc = psutil.Process()
    open_file_path = []
    open_files = proc.open_files()
    for o_file in open_files:
        open_file_path.append(o_file[0])    # get the path

    return open_file_path

def get_all_processes_openfiles(proc_name_contain_str=None):
    # check all open files
    import getpass
    user_name = getpass.getuser()
    all_open_files = []
    for proc in psutil.process_iter():
        try:
            # _proc = proc.as_dict(attrs=['cpu_times', 'name', 'pid', 'status'])
            # print(proc)
            if proc.username() != user_name:
                continue
            if proc_name_contain_str is not None:
                if proc_name_contain_str not in proc.name():
                    continue
            if proc.is_running() is False:
                continue
            open_files = proc.open_files()
            open_file_path = []
            for o_file in open_files:
                open_file_path.append(o_file[0])
                all_open_files.append(o_file[0])
            print(proc.pid, proc.name(), proc.username(), 'open %d files'%len(open_file_path))   # proc.is_running()

        except psutil.NoSuchProcess:
            continue
        except psutil.ZombieProcess: #
            continue
        except:
            print('unknown except')
            continue


        # # print('process: id, name, started, open %d files'%len(open_file_path), proc[0], proc[1], proc[2])
    return all_open_files

if __name__=='__main__':
    pass
