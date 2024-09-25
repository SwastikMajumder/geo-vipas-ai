import streamlit as st
import engine_r # Assuming this function is available in engine_r
from PIL import Image, ImageDraw  # For creating the initial black-and-white image
import engine

# Function to create a black-and-white placeholder image
def create_placeholder_image():
    image = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((200, 240), "Placeholder", fill='black')  # Optional placeholder text
    return image

# Initialize session state for accumulated commands and initial image
if 'accumulated_commands' not in st.session_state:
    st.session_state.accumulated_commands = ""  # String to accumulate commands

if 'diagram_image' not in st.session_state:
    # Initialize with a black-and-white placeholder image
    st.session_state.diagram_image = create_placeholder_image()

if 'num_points' not in st.session_state:
    st.session_state.num_points = 0  # To store the number of points for the join command

if 'lines' not in st.session_state:
    st.session_state.lines = []  # To store lines returned by the engine

if 'angles' not in st.session_state:
    st.session_state.angles = []  # To store angles returned by the engine

# Function to run geometry commands with accumulated command string
def run_geometry_command(command):
    # Prepare the full command string by appending the new command
    new_accumulated_commands = f"{st.session_state.accumulated_commands}\n{command}".strip()
    result = engine_r.run_parallel_function(new_accumulated_commands)  # Send the whole string
    if result == "error":
        return None, "error"
    else:
        # Update the accumulated commands only if the command is successful
        st.session_state.accumulated_commands = new_accumulated_commands

        # Assuming result is a tuple: (image, num_points, line list, angle list)
        diagram_image = result[0]  # Image of the diagram
        num_points = result[1]  # Number of points
        line_list = result[2]  # List of lines
        angle_list = result[3]  # List of angles

        # Update session state
        st.session_state.num_points = num_points
        st.session_state.lines = line_list  # Store the lines for further commands
        st.session_state.angles = angle_list  # Store the angles for further commands
        
        return {
            "image": diagram_image,
            "num_points": num_points,
            "line_list": line_list,
            "angle_list": angle_list
        }, None

# Streamlit app interface
st.title("Geometry Ai")

# Sidebar with tool selection
st.sidebar.title("Toolbox")
tool = st.sidebar.radio("Select a tool:", ("Draw", "Perpendicular", "Join", "Extend", "Split", "Set Angles Equal", "Angle Value", "Set Lines Equal", "Parallel Line"))

st.sidebar.subheader("Accumulated Commands")
st.sidebar.text_area("Commands", value=st.session_state.accumulated_commands, height=200)


# Tool: Draw (Triangle or Quadrilateral)
if tool == "Draw":
    draw_option = st.radio("What would you like to draw?", ("Draw Triangle", "Draw Quadrilateral"))
    command = "draw triangle" if draw_option == "Draw Triangle" else "draw quadrilateral"

    if st.button("Run Command"):
        result, error = run_geometry_command(command)
        if error:
            st.error("Invalid command. Please try again.")
        else:
            if result["image"]:
                st.session_state.diagram_image = result["image"]

            if result["num_points"]:
                st.write(f"Number of Points: {result['num_points']}")
            if result["line_list"]:
                st.write("Line List:")
                st.write(result["line_list"])
            if result["angle_list"]:
                st.write("Angle List:")
                st.write(result["angle_list"])
            

# Tool: Join
if tool == "Join":
    if st.session_state.num_points > 1:
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Generates ['A', 'B', 'C', ...]
        point1 = st.selectbox("Select the first point to join:", points)
        point2 = st.selectbox("Select the second point to join:", points)

        if st.button("Join Points"):
            if point1 != point2:  # Ensure different points are selected
                command = f"join {point1}{point2}"
                result, error = run_geometry_command(command)

                if error:
                    st.error("Invalid command. Please try again.")
                else:
                    if result["image"]:
                        st.session_state.diagram_image = result["image"]

                    if result["num_points"]:
                        st.write(f"Number of Points: {result['num_points']}")
                    if result["line_list"]:
                        st.write("Line List:")
                        st.write(result["line_list"])
                    if result["angle_list"]:
                        st.write("Angle List:")
                        st.write(result["angle_list"])
            else:
                st.error("Please select two different points to join.")
    else:
        st.warning("At least two points are needed to perform a join operation.")

# Tool: Extend
if tool == "Extend":
    if st.session_state.num_points > 0 and st.session_state.lines:
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Points A, B, C, ...
        lines = st.session_state.lines  # List of lines returned by the engine

        # Select a point for extending
        point = st.selectbox("Select a point to extend from:", points)
        
        # Select a line to extend
        line = st.selectbox("Select a line to extend:", lines)

        # Slider for the extension length
        extension_length = st.slider("Extension Length (units):", min_value=1, max_value=200, value=100)

        if st.button("Extend Line"):
            command = f"extend {line} from {point} for {extension_length}"
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one point and one line are needed to perform an extend operation.")

# Tool: Perpendicular
if tool == "Perpendicular":
    if st.session_state.lines:
        lines = st.session_state.lines  # List of lines returned by the engine
        line_to_perpendicular = st.selectbox("Select a line to create a perpendicular:", lines)

        # Select a point for the perpendicular
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Points A, B, C, ...
        point = st.selectbox("Select a point to create the perpendicular from:", points)

        if st.button("Create Perpendicular"):
            command = f"perpendicular {point} to {line_to_perpendicular}"
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to perform a perpendicular operation.")


# Tool: Split
if tool == "Split":
    if st.session_state.lines:
        lines = st.session_state.lines  # List of lines returned by the engine
        line_to_split = st.selectbox("Select a line to split:", lines)

        if st.button("Split Line"):
            command = f"split {line_to_split}"
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to perform a split operation.")

# Tool: Set Angles Equal
if tool == "Set Angles Equal":
    if st.session_state.angles and len(st.session_state.angles) > 1:
        angles = [f"∠{angle}" for angle in st.session_state.angles]  # Formatting angles
        angle1 = st.selectbox("Select the first angle to set equal:", angles)[1:]
        angle2 = st.selectbox("Select the second angle to set equal:", angles)[1:]

        if st.button("Set Angles Equal"):
            if angle1 != angle2:  # Ensure different angles are selected
                command = f"equation angle_eq {angle1} {angle2}"
                result, error = run_geometry_command(command)

                if error:
                    st.error("Invalid command. Please try again.")
                else:
                    if result["image"]:
                        st.session_state.diagram_image = result["image"]

                    if result["num_points"]:
                        st.write(f"Number of Points: {result['num_points']}")
                    if result["line_list"]:
                        st.write("Line List:")
                        st.write(result["line_list"])
                    if result["angle_list"]:
                        st.write("Angle List:")
                        st.write(result["angle_list"])
            else:
                st.error("Please select two different angles to set equal.")
    else:
        st.warning("At least two angles are needed to perform the set equal operation.")

# Tool: Angle Value
if tool == "Angle Value":
    if st.session_state.angles and len(st.session_state.angles) > 0:
        angles = [f"∠{angle}" for angle in st.session_state.angles]  # Formatting angles
        angle = st.selectbox("Select an angle to set value:", angles)[1:]

        angle_value = st.number_input("Enter the angle value (degrees):", min_value=0, max_value=360)

        if st.button("Set Angle Value"):
            command = f"equation angle_val {angle} {angle_value}"
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least two lines are needed to perform the set equal operation.")

# Tool: Line Equality
if tool == "Set Lines Equal":
    if st.session_state.lines and len(st.session_state.lines) > 1:
        lines = st.session_state.lines  # List of lines returned by the engine
        line1 = st.selectbox("Select the first line:", lines)
        line2 = st.selectbox("Select the second line:", lines)

        if st.button("Set Lines Equal"):
            command = f"equation line_eq {line1} {line2}"  # Assuming this command checks if lines are equal
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least two lines are needed to check equality.")



# Tool: Parallel Line
if tool == "Parallel Line":
    if st.session_state.lines and st.session_state.num_points > 0:
        lines = st.session_state.lines  # List of lines returned by the engine
        line1 = st.selectbox("Select the first line (for reference):", lines)
        line2 = st.selectbox("Select the second line (to be parallel to):", lines)

        if st.button("Create Parallel Line"):
            command = f"equation parallel_line {line2} {line1}"
            result, error = run_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to create a parallel line.")

# Add button to run accumulated commands
if st.sidebar.button("Execute Accumulated Commands"):
    command = st.session_state.accumulated_commands  # Get the accumulated commands
    if command:  # Ensure there are commands to execute
        output = engine.run_parallel_function(command + "\ncompute\ncompute")  # Run the accumulated commands
        st.subheader("Analysis:")
        st.markdown("```\n" + output + "\n```") 
    else:
        st.warning("No commands to execute.")

# Display the diagram image
st.image(st.session_state.diagram_image, caption="Geometry Diagram", use_column_width=True)
