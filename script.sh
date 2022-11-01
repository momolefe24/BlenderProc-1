#Environment
file_dir="examples/basics/camera_sampling"
scene_dir="examples/resources"
output_dir="examples/basics/camera_sampling/output"
install_dir="./"
args=("$@")

# blenderproc run examples/basics/camera_sampling/main.py examples/resources/scene.obj examples/basics/camera_sampling/output --blender-install-path ./

while getopts f:s:Z opt
do
  case $opt in
    f) filename="${file_dir}/$OPTARG";;
    s) scene_object="${scene_dir}/$OPTARG";;
    Z) z="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

# ./script.sh -f main.py -s marina2.blend -R False
echo "filename: ${filename}"
echo "scene_object: ${scene_object}"
echo "random_sample: ${random_sample}"
if [ -f $filename -a -f $scene_object ] 
then
if [ ! -d $output ]
then 
    mkdir -p $output 
fi

blenderproc run $filename $scene_object $output_dir --blender-install-path ./

else
echo "files do not exist"
fi
