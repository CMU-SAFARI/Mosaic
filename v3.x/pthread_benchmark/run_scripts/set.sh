#!/bin/sh

mylist=`cat apps.txt`
set -- $mylist

for APP1; do
	    echo  "./gpgpu_ptx_sim__mergedapps -sing" $APP1 \> output\_$APP1.txt>  mainscript\_$APP1
		echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 \> output\_$APP1\_$APP1.txt>  mainscript\_$APP1\_$APP1
		echo  "./gpgpu_ptx_sim__mergedapps -sing0" $APP1 \> output\_$APP1\_NoStream.txt>  mainscript\_$APP1\_NoStream
        shift
        for APP2; do
                echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2

		done
done

mylist2=`cat apps2.txt`
set -- $mylist2

for APP1; do
    echo  "./gpgpu_ptx_sim__mergedapps -sing" $APP1 \> output\_$APP1.txt>  mainscript\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 \> output\_$APP1\_$APP1.txt>  mainscript\_$APP1\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -sing0" $APP1 \> output\_$APP1\_NoStream.txt>  mainscript\_$APP1\_NoStream
    shift
    for APP2; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2

    done
done

mylist3=`cat apps3.txt`
set -- $mylist3

for APP1; do
    echo  "./gpgpu_ptx_sim__mergedapps -sing" $APP1 \> output\_$APP1.txt>  mainscript\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 \> output\_$APP1\_$APP1.txt>  mainscript\_$APP1\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -sing0" $APP1 \> output\_$APP1\_NoStream.txt>  mainscript\_$APP1\_NoStream
    shift
    for APP2; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2

    done
done

mylist4=`cat apps4.txt`
set -- $mylist4

for APP1; do
    echo  "./gpgpu_ptx_sim__mergedapps -sing" $APP1 \> output\_$APP1.txt>  mainscript\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 \> output\_$APP1\_$APP1.txt>  mainscript\_$APP1\_$APP1
    echo  "./gpgpu_ptx_sim__mergedapps -sing0" $APP1 \> output\_$APP1\_NoStream.txt>  mainscript\_$APP1\_NoStream
    shift
    for APP2; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2

    done
done


for APP1 in $mylist; do
    for APP2 in $mylist2; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done
for APP1 in $mylist; do
    for APP2 in $mylist3; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done
for APP1 in $mylist; do
    for APP2 in $mylist4; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done
for APP1 in $mylist2; do
    for APP2 in $mylist3; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done
for APP1 in $mylist2; do
    for APP2 in $mylist4; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done
for APP1 in $mylist3; do
    for APP2 in $mylist4; do
        echo  "./gpgpu_ptx_sim__mergedapps -apps" $APP1 $APP2 \> output\_$APP1\_$APP2.txt> mainscript\_$APP1\_$APP2
    done
    shift
done