/**
 * environment: 
 * gcc 11.2.0
**/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>

#include "omp.h"

#define PRINT 2
#define TRUE 1
#define FALSE 0

#define BIGGER 1
#define SMALLER -1
#define EQUAL 0

#define COORDINATE_MESSAGE 1
#define NOT_ELECTED 0

#define NUM_CONFIGURATION 3


// Node of linked list
typedef struct queue_node{
    int node;
    struct queue_node *next;
}Queue_Node;

// Priority Queue
typedef struct queue{
    Queue_Node *head;
    Queue_Node *last;
    unsigned int length;
}Queue;

// Coordinate
typedef struct coordinate{
    int x;
    int y;
}Coordinate;


// return the minimum of two
int min(int a, int b);

//===========================================================
//adjacent matrix
/**
 * read adjacent matrix from input
 * return adjacent matrix
 */
int **read_adjacent_matrix(int *node_number, int *edge_number, char *filename);
/**
 * copy an adjacent matrix
 */
int **copy_adjacent_matrix(int node_number,int** adjacent_matrix);
/**
 * create empty adjacent matrix
 */
int **create_adjacent_matrix(int node_number);
//free the adjacent matrix
void free_adjacent_matrix(int node_number, int** adjacent_matrix);
//print the adjacent matrix to stdout
void print_adjacent_matrix(int node_number, int** adjacent_matrix);

//===========================================================
//queue

// create an empty priority queue
Queue *create_queue();
// create a new node
Queue_Node *new_node(int node);
// whether the queue is empty
int isEmpty(Queue *queue);
// free a priority queue and data inside
void free_queue(Queue *queue);
// push a node to the tail of queue
void push(Queue *queue, int node);
// pop the first element and return node
int pop(Queue *queue);

//===========================================================
//sequential algorithm
void sequential_vertex_cover(int node_number,int edge_number, int** adjacent_matrix);
//parallel algorithm
void parallel_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount);
//mpi parallel algorithm
void mpi_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount);
//ring based mpi parallel algorithm
void ring_based_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount);

//===========================================================
/**
 * increase the subset by n
 * return FALSE if reach the end of the subset
*/ 
int increment_n(int *vertex_set, int subset_vertex_number, int node_number, int n);
// increase the subset by 1
int increment(int *vertex_set, int subset_vertex_number, int node_number);
/**
 * verify whether the subset is a vlid vertex cover
*/
int valid_vertex_cover(int node_number,int edge_number, int** adjacent_matrix,
                        Coordinate *edges,
                        int *subset, int subset_node_number);
/**
 * verify whether the input matrix cover every node
*/
int verify(Coordinate *edges, int edge_number, int **edge_covered);
/**
 * update the covered edge
*/
void update_covered_edge(int added_node,int node_number, int** adjacent_matrix, int **edge_covered);
/**
 * compare the ring info
 * if a > b, return BIGGER (1)
 * if b > a, return SMALLER (-1)
 * if a == b, return EQUAL (0)
*/
int compare_ring_info(int* a, int* b);


// read
int main(int argc, char * argv[]){

    // initialize mpi
    MPI_Init(&argc, &argv);

    int node_number,edge_number,is_parallel,number_thread;
    int **adjacent_matrix;

    if(argc != NUM_CONFIGURATION + 1){
        perror("Not enough configuration!");
        exit(1);
    }

    is_parallel = atoi(argv[1]);
    number_thread = atoi(argv[2]);

    // scan the input
    adjacent_matrix = read_adjacent_matrix(&node_number,&edge_number, argv[3]);

    // do not print when there are multiple nodes
    if(is_parallel < 2){
        // print number of node and edge
        printf("%d %d ",node_number,edge_number);
    }

    // set number of threads
    omp_set_num_threads(number_thread);

    // computation
    if(is_parallel == 0){
        sequential_vertex_cover(node_number,edge_number,adjacent_matrix);
    }else if(is_parallel == 1){
        parallel_vertex_cover(node_number,edge_number,adjacent_matrix, 
                        number_thread, number_thread);
    }else if(is_parallel == 2){
        mpi_vertex_cover(node_number,edge_number,adjacent_matrix, 
                        number_thread, number_thread);
    }else if(is_parallel == 3){
        ring_based_vertex_cover(node_number,edge_number,adjacent_matrix, 
                        number_thread, number_thread);
    }else{
        perror("is_parallel wrong!");
        exit(1);
    }

    //free the matrix
    free_adjacent_matrix(node_number,adjacent_matrix);

    // finish MPI
    MPI_Finalize();

    return 0;

}

// return the minimum of two
int min(int a, int b){
    if(a < b){
        return a;
    }
    return b;
}

/**
 * read adjacent matrix from input
 * return adjacent matrix
 */
int **read_adjacent_matrix(int *node_number, int *edge_number, char *filename){
    // store input edge u->v with capactiy
    int u,v,current_edge_number = 0;
    int **adjacent_matrix;
    int i;
    FILE *input;

    input = fopen(filename, "r");

    //scan the number of node
    fscanf(input, "%d\n", node_number);

    adjacent_matrix = create_adjacent_matrix(*node_number);

    //read the edges
    while(fscanf(input, "%d %d\n", &u,&v) == 2){
        adjacent_matrix[u][v] = 1;
        adjacent_matrix[v][u] = 1;
        current_edge_number++;
    }

    *edge_number = current_edge_number;

    return adjacent_matrix;

}

/**
 * copy an adjacent matrix
 */
int **copy_adjacent_matrix(int node_number,int** adjacent_matrix){
    int i,j;
    int **new_adjacent_matrix;

    //allocate the adjacent matrix (use malloc to save time)
    new_adjacent_matrix = (int **)(malloc(sizeof(int*) * (node_number)));
    for(i=0;i<node_number;i++){
        new_adjacent_matrix[i] = (int *)(malloc(node_number *sizeof(int)));
    }

    for(i=0;i<node_number;i++){
        for(j=0;j<node_number;j++){
            new_adjacent_matrix[i][j] = adjacent_matrix[i][j];
        }
    }

    return new_adjacent_matrix;
}

/**
 * create empty adjacent matrix
 */
int **create_adjacent_matrix(int node_number){
    int i;
    int **adjacent_matrix;

    //allocate the adjacent matrix
    adjacent_matrix = (int **)(malloc(sizeof(int*) * (node_number)));
    for(i=0;i<node_number;i++){
        adjacent_matrix[i] = (int *)(calloc(node_number,sizeof(int)));
    }

    return adjacent_matrix;

}

//free the adjacent matrix
void free_adjacent_matrix(int node_number, int** adjacent_matrix){

    int i;
    for(i=0;i<node_number;i++){
        free(adjacent_matrix[i]);
    }
    free(adjacent_matrix);

}

//print the adjacent matrix to stdout
void print_adjacent_matrix(int node_number, int** adjacent_matrix){

    int i,j;
    for(i=0;i<node_number;i++){
        for(j=0;j<node_number;j++){
            printf("%d ",adjacent_matrix[i][j]);
        }
        printf("\n");
    }

}

// create an empty priority queue
Queue *create_queue(){
    Queue *temp = (Queue *)malloc(sizeof(Queue));
    temp->head = NULL;
    temp->last = NULL;
    temp->length = 0;
    return temp;
}

// create a new node
Queue_Node *new_node(int node){
    Queue_Node *temp = (Queue_Node *)malloc(sizeof(Queue_Node));
    temp->node = node;
    temp->next = NULL;
 
    return temp;
}

// whether the queue is empty
int isEmpty(Queue *queue){
    return queue->length == 0;
}

// free a priority queue and data inside
void free_queue(Queue *queue){
    if(queue == NULL){
        return;
    }
    Queue_Node *curr_node=queue->head;
    Queue_Node *temp_node;
    while(curr_node!=NULL){
        temp_node = curr_node;
        curr_node = curr_node->next;
        free(temp_node);
    }
    free(queue);
}

// push a node to the tail of queue
void push(Queue *queue, int node){

    Queue_Node *Qnode;

    //create new queue node
    Qnode = new_node(node);
    
    if(queue->head == NULL){
        //if it is the first node
        queue->head = Qnode;
        queue->last = Qnode;
    }else{
        //add it to the tail of queue
        queue->last->next = Qnode;
        queue->last = Qnode;
    }
    (queue->length)++;

}

// pop the first element and return node
int pop(Queue *queue){

    int node;

    if(queue->head==NULL){
        printf("ERROR: try to pop from an empty queue!");
        return -1;
    }

    Queue_Node *temp = queue->head;
    node = temp->node;
    queue->head = queue->head->next;
    free(temp);
    (queue->length)--;

    //queue is empty
    if(queue->head == NULL){
        queue->last = NULL;
    }
    
    return node;
}

//find the minimal set of vertices that cover all the edges
void sequential_vertex_cover(int node_number,int edge_number, int** adjacent_matrix){
    //number of vertices in the set
    int number_vertices;
    int i,j,k,current_edges=0;
    // whether found the minial vertex cover
    int has_found = FALSE;

    //coordinate of edge
    Coordinate temp_coord;

    double start_time = omp_get_wtime();

    //record the current vertex set
    int *vertex_set = (int*)malloc(sizeof(int) * node_number);

    //store the coordinates of edges into a dynamic array
    Coordinate *edges = (Coordinate *)malloc(sizeof(Coordinate) * edge_number);
    for(i=0;i<node_number;i++){
        for(j=i;j<node_number;j++){
            if(adjacent_matrix[i][j] != 0){
                temp_coord.x = i;
                temp_coord.y = j;
                edges[current_edges] = temp_coord;
                current_edges++;
            }
        }
    }

    // loop throught all the vertices set to find minimal vertex cover
    for(number_vertices=1;number_vertices<=node_number;number_vertices++){

        // init the vertex set
        for(i=0;i<number_vertices;i++){
            vertex_set[i] = i;
        }
        // prepare for increment
        vertex_set[number_vertices-1] -= 1;

        while(increment(vertex_set, number_vertices, node_number)){

            if(valid_vertex_cover(node_number,edge_number, adjacent_matrix,edges,
                                    vertex_set, number_vertices)){
                has_found = TRUE;
                break;
            }

        }

        if(has_found){
            break;
        }

    }

    double end_time = omp_get_wtime();

    //print time
    printf("%f ",end_time-start_time);

    // print the result
    printf("%d ", number_vertices);
    for(i=0;i<number_vertices;i++){
        printf("%d ", vertex_set[i]);
    }
    printf("\n");

    //free the dynamic resources
    free(vertex_set);
    free(edges);
}

//find the minimal set of vertices that cover all the edges
void parallel_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount){
    //number of vertices in the set
    int number_vertices;
    int i,j,k,current_edges=0;
    // whether found the minial vertex cover
    int has_found = FALSE;

    //coordinate of edge
    Coordinate temp_coord;

    double start_time = omp_get_wtime();

    //record the current vertex set
    int *correct_vertex_set = NULL;

    //store the coordinates of edges into a dynamic array
    Coordinate *edges = (Coordinate *)malloc(sizeof(Coordinate) * edge_number);
    for(i=0;i<node_number;i++){
        for(j=i;j<node_number;j++){
            if(adjacent_matrix[i][j] != 0){
                temp_coord.x = i;
                temp_coord.y = j;
                edges[current_edges] = temp_coord;
                current_edges++;
            }
        }
    }

    // loop throught all the vertices set to find minimal vertex cover
    for(number_vertices=1;number_vertices<=node_number;number_vertices++){

        #pragma omp parallel for num_threads(number_thread) shared(has_found, correct_vertex_set)
        for(i=0;i<skip_amount;i++){

            //record the current vertex set
            int *vertex_set = (int*)malloc(sizeof(int) * node_number);
            //whether the current thread found the valid vertex cover
            int I_found = FALSE;

            // init the vertex set
            for(j=0;j<number_vertices;j++){
                vertex_set[j] = j;
            }
            // prepare for increment
            vertex_set[number_vertices-1] -= 1;
            
            // start with different init
            increment_n(vertex_set, number_vertices, node_number, i);
            
            // do the computation
            while(increment_n(vertex_set, number_vertices, node_number, skip_amount)){
                if(valid_vertex_cover(node_number,edge_number, adjacent_matrix,edges,
                                        vertex_set, number_vertices)){
                    I_found = TRUE;
                    has_found = TRUE;
                    break;
                }

                // other thread might found valid vertex cover
                if(has_found){
                    break;
                }
            }

            if(I_found){
                if(correct_vertex_set == NULL){
                    correct_vertex_set = vertex_set;
                }else{
                    free(vertex_set);
                }
            }else{
                free(vertex_set);
            }

        }

        // found the valid vertex cover
        if(has_found){
            break;
        }
        
    }

    double end_time = omp_get_wtime();

    //print time
    printf("%f ",end_time-start_time);

    // print the result
    printf("%d ", number_vertices);
    for(i=0;i<number_vertices;i++){
        printf("%d ", correct_vertex_set[i]);
    }
    printf("\n");

    //free the dynamic resources
    free(correct_vertex_set);
    free(edges);
}

//find the minimal set of vertices that cover all the edges
void mpi_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount){

    // mpi objects 
    int world_rank, world_size;
    // store whether the corresponding node find the vertex cover
    int *has_found_node;

    //number of vertices in the set
    int number_vertices;
    int i,j,k,current_edges=0;
    // whether found the minial vertex cover
    int has_found = FALSE;

    //coordinate of edge
    Coordinate temp_coord;

    double start_time = omp_get_wtime();

    //get the rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    has_found_node = (int*)malloc(sizeof(int) * world_size);

    //update skip amount
    skip_amount *= world_size;

    //record the current vertex set
    int *correct_vertex_set = NULL;

    //store the coordinates of edges into a dynamic array
    Coordinate *edges = (Coordinate *)malloc(sizeof(Coordinate) * edge_number);
    for(i=0;i<node_number;i++){
        for(j=i;j<node_number;j++){
            if(adjacent_matrix[i][j] != 0){
                temp_coord.x = i;
                temp_coord.y = j;
                edges[current_edges] = temp_coord;
                current_edges++;
            }
        }
    }

    // loop throught all the vertices set to find minimal vertex cover
    for(number_vertices=1;number_vertices<=node_number;number_vertices++){

        #pragma omp parallel for num_threads(number_thread) shared(has_found, correct_vertex_set)
        for(i=world_rank;i<skip_amount;i+=world_size){

            //record the current vertex set
            int *vertex_set = (int*)malloc(sizeof(int) * node_number);
            //whether the current thread found the valid vertex cover
            int I_found = FALSE;

            // init the vertex set
            for(j=0;j<number_vertices;j++){
                vertex_set[j] = j;
            }
            // prepare for increment
            vertex_set[number_vertices-1] -= 1;
            
            // start with different init
            increment_n(vertex_set, number_vertices, node_number, i);
            
            // do the computation
            while(increment_n(vertex_set, number_vertices, node_number, skip_amount)){
                if(valid_vertex_cover(node_number,edge_number, adjacent_matrix,edges,
                                        vertex_set, number_vertices)){
                    I_found = TRUE;
                    has_found = TRUE;
                    break;
                }

                // other thread might found valid vertex cover
                if(has_found){
                    break;
                }
            }

            if(I_found){
                if(correct_vertex_set == NULL){
                    correct_vertex_set = vertex_set;
                }else{
                    free(vertex_set);
                }
            }else{
                free(vertex_set);
            }

        }

        // every process need to complete its work on current number of subset
        // find whether a node find the answer
        MPI_Gather(&has_found, 1, MPI_INT, has_found_node, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // check whether the vertex cover is found and decide which proces print the result
        if(world_rank == 0){
            //whether the vertex cover has been found
            for(i=0;i<world_size;i++){
                if(has_found_node[i]){
                    has_found = TRUE;
                    break;
                }
            }
            // tell every node the vertex cover if vertex cover has been found 
            if(has_found){
                for(j=0;j<world_size;j++){
                    has_found_node[j] = TRUE;
                }
                // tell the node who found it to print the result
                has_found_node[i] = PRINT;
            }
        }

        // the process 0 broadcast whether the process found the answer
        // and which process print the output
        MPI_Scatter(has_found_node, 1, MPI_INT, &has_found, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // found the valid vertex cover
        if(has_found){
            break;
        }
        
    }

    double end_time = omp_get_wtime();

    // the process is response for print the result
    if(has_found == PRINT){

        // print number of node and edge
        printf("%d %d ",node_number,edge_number);

        //print time
        printf("%f ",end_time-start_time);

        // print the result
        printf("%d ", number_vertices);
        for(i=0;i<number_vertices;i++){
            printf("%d ", correct_vertex_set[i]);
        }
        printf("\n");
    }

    

    //free the dynamic resources
    free(correct_vertex_set);
    free(edges);
    
}

//find the minimal set of vertices that cover all the edges
void ring_based_vertex_cover(int node_number,int edge_number, int** adjacent_matrix, 
                        int number_thread, int skip_amount){

    // mpi objects 
    int world_rank, world_size;
    // whether receive a message
    int flag;
    MPI_Status status;

    //number of vertices in the set
    int number_vertices;
    int i,j,k,current_edges=0;
    // whether found the minial vertex cover
    int has_found = FALSE;
    // whether the process is the leader
    int leader = FALSE;
    // whether the process blocks a ring data
    int block = FALSE;
    // whether the election is finished
    int election_finished = FALSE;
    // whether other process has found the vertex cover
    int other_has_found = FALSE;
    // whether the process hold a message
    int hold = FALSE;

    //coordinate of edge
    Coordinate temp_coord;

    double start_time = omp_get_wtime();

    //get the rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ring data 
    // (store the rank of process who get vertex cover, the size of subset and 
    // whether this data is a elected command)
    // compare size of subset then rank of process
    int *temp_ring_data = (int *)malloc(sizeof(int) * 3);
    temp_ring_data[1] = node_number + 2;
    int *current_ring_data = (int *)malloc(sizeof(int) * 3);
    current_ring_data[1] = node_number + 1;


    // Declare the buffer and attach it
    int buffer_attached_size = MPI_BSEND_OVERHEAD + sizeof(int)*3*world_size;
    int* buffer_attached = (int*)malloc(buffer_attached_size);
    MPI_Buffer_attach(buffer_attached, buffer_attached_size);

    //update skip amount
    skip_amount *= world_size;

    //record the current vertex set
    int *correct_vertex_set = NULL;

    //store the coordinates of edges into a dynamic array
    Coordinate *edges = (Coordinate *)malloc(sizeof(Coordinate) * edge_number);
    for(i=0;i<node_number;i++){
        for(j=i;j<node_number;j++){
            if(adjacent_matrix[i][j] != 0){
                temp_coord.x = i;
                temp_coord.y = j;
                edges[current_edges] = temp_coord;
                current_edges++;
            }
        }
    }

    // loop throught all the vertices set to find minimal vertex cover
    // work until exhausted all the possible subset with size under 
    // the vertex set which has been found
    for(number_vertices=1;number_vertices<current_ring_data[1];number_vertices++){

        #pragma omp parallel for num_threads(number_thread) shared(has_found, correct_vertex_set, flag, status, temp_ring_data, current_ring_data, hold, other_has_found, buffer_attached)
        for(i=world_rank;i<skip_amount;i+=world_size){

            //record the current vertex set
            int *vertex_set = (int*)malloc(sizeof(int) * node_number);
            //whether the current thread found the valid vertex cover
            int I_found = FALSE;

            // init the vertex set
            for(j=0;j<number_vertices;j++){
                vertex_set[j] = j;
            }
            // prepare for increment
            vertex_set[number_vertices-1] -= 1;
            
            // start with different init
            increment_n(vertex_set, number_vertices, node_number, i);
            
            // do the computation
            while(increment_n(vertex_set, number_vertices, node_number, skip_amount)){

                // only the master check whether there is a message
                if(omp_get_thread_num() == 0){
                    //whether receive a message
                    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
                    if(flag){
                        MPI_Recv(temp_ring_data, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                        // accept the data if it is smaller than current
                        if(compare_ring_info(temp_ring_data, current_ring_data) == SMALLER){
                            current_ring_data[0] = temp_ring_data[0];
                            current_ring_data[1] = temp_ring_data[1];
                            current_ring_data[2] = temp_ring_data[2];

                            // check whether the process exhausted all the possible 
                            // subset with size under the vertex set
                            if(number_vertices >= current_ring_data[1]){
                                // send this ring data to next process
                                hold = FALSE;
                                other_has_found = TRUE;
                                MPI_Bsend(current_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
                            }else{
                                //hold the message until it found the smaller or
                                // exhausted all the possible subset with size 
                                // under the vertex set 
                                hold = TRUE;
                            }
                        }
                    }
                }

                if(valid_vertex_cover(node_number,edge_number, adjacent_matrix,edges,
                                        vertex_set, number_vertices)){
                    I_found = TRUE;
                    has_found = TRUE;
                    break;
                }

                // other thread might found valid vertex cover
                if(has_found || other_has_found){
                    break;
                }
            }

            if(I_found){
                if(correct_vertex_set == NULL){
                    correct_vertex_set = vertex_set;
                }else{
                    free(vertex_set);
                }
            }else{
                free(vertex_set);
            }

        }       

        // found the valid vertex cover
        if(has_found && !other_has_found){
            // hold the smaller ring data
            temp_ring_data[0] = world_rank;
            temp_ring_data[1] = number_vertices;
            if(compare_ring_info(temp_ring_data, current_ring_data) == SMALLER){
                // prepare the election message
                current_ring_data[0] = world_rank;
                current_ring_data[1] = number_vertices;
                current_ring_data[2] = NOT_ELECTED;
                // participate the election
                hold = FALSE;
                // send out election message to next process
                MPI_Bsend(current_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
            }
            break;
        }
        if(other_has_found){
            break;
        }
        
    }

    if(hold){
        MPI_Bsend(current_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
    }

    //Ring-Based Election
    while(!election_finished){
        MPI_Recv(temp_ring_data, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        // this process becomes leader or all processes know it is leader
        // if receive a message initially send by itself
        if(temp_ring_data[0] == world_rank){
            // finish the elction if the message is a coordinate message
            if(temp_ring_data[2]){
                election_finished =TRUE;
            }else{
                //become leader and send a coordinate message
                leader = TRUE;
                temp_ring_data[2] = COORDINATE_MESSAGE;
                MPI_Send(temp_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
            }
        }else if(temp_ring_data[2] == COORDINATE_MESSAGE){
            // receive the coordinate message and 
            election_finished =TRUE;
            MPI_Send(temp_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
        }
        // only send smaller message
        else if(compare_ring_info(temp_ring_data, current_ring_data) == SMALLER){
            current_ring_data[0] = temp_ring_data[0];
            current_ring_data[1] = temp_ring_data[1];
            current_ring_data[2] = temp_ring_data[2];
            MPI_Send(current_ring_data, 3, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD);
        }
    }
    

    double end_time = omp_get_wtime();

    // the process is response for print the result
    if(leader == TRUE){

        // print number of node and edge
        printf("%d %d ",node_number,edge_number);

        //print time
        printf("%f ",end_time-start_time);

        // print the result
        printf("%d ", number_vertices);
        for(i=0;i<number_vertices;i++){
            printf("%d ", correct_vertex_set[i]);
        }
        printf("\n");
    }

    

    //free the dynamic resources
    free(correct_vertex_set);
    free(temp_ring_data);
    free(current_ring_data);
    // Detach the buffer. It blocks until all messages stored are sent.
    MPI_Buffer_detach(&buffer_attached, &buffer_attached_size);
    free(buffer_attached);
    free(edges);
    
}

/**
 * increase the subset by n
 * return FALSE if reach the end of the subset
*/ 
int increment_n(int *vertex_set, int subset_vertex_number, int node_number, int n){
    // remain the same
    if(n==0){
        return TRUE;
    }

    int i;
    for(i=0;i<n;i++){
        // reach the end of the subset
        if(!increment(vertex_set, subset_vertex_number, node_number)){
            return FALSE;
        }
    }

    return TRUE;

}

/**
 * increase the subset by 1
 * return FALSE if reach the end of the subset
*/ 
int increment(int *vertex_set, int subset_vertex_number, int node_number){
    // whether this is the last subset
    int has_next = FALSE;
    int i,k;
    int next_number, 
        range_max; // the max number a position can have

    // find next number to increment 
    for(k=subset_vertex_number-1;k>=0;k--){
        next_number = vertex_set[k] + 1;
        range_max = node_number - subset_vertex_number + k + 1;
        if(next_number < range_max){
            vertex_set[k] += 1;
            has_next = TRUE;
            break;
        }
    }

    for(i=k+1;i<subset_vertex_number;i++){
        vertex_set[i] = vertex_set[i-1] + 1;
    }

    return has_next;

}

/**
 * verify whether the subset is a vlid vertex cover
*/
int valid_vertex_cover(int node_number,int edge_number, int** adjacent_matrix,
                        Coordinate *edges,
                        int *subset, int subset_node_number){
    int i, result;
    int **edge_covered = create_adjacent_matrix(node_number);

    for(i=0;i<subset_node_number;i++){
        update_covered_edge(subset[i], node_number, adjacent_matrix, edge_covered);
    }

    result = verify(edges, edge_number, edge_covered);

    // release resource
    free_adjacent_matrix(node_number, edge_covered);

    return result;

}

/**
 * verify whether the input matrix cover every node
*/
int verify(Coordinate *edges, int edge_number, int **edge_covered){
    int i,x,y;

    // check whether each edge is covered
    for(i=0;i<edge_number;i++){
        x = edges[i].x;
        y = edges[i].y;
        //not covered
        if(!edge_covered[x][y]){
            return FALSE;
        }
    }

    return TRUE;

}

/**
 * update the covered edge
*/
void update_covered_edge(int added_node,int node_number, int** adjacent_matrix, int **edge_covered){
    int i;
    for(i=0;i<node_number;i++){
        edge_covered[i][added_node] = edge_covered[i][added_node] || adjacent_matrix[i][added_node];
        edge_covered[added_node][i] = edge_covered[added_node][i] || adjacent_matrix[added_node][i];
    }

}

/**
 * compare the ring info
 * if a > b, return BIGGER (1)
 * if b > a, return SMALLER (-1)
 * if a == b, return EQUAL (0)
*/
int compare_ring_info(int* a, int* b){
    if(a[1] > b[1]){
        return BIGGER;
    }else if(a[1] < b[1]){
        return SMALLER;
    }else{
        if(a[0] > b[0]){
        return BIGGER;
        }else if(a[0] < b[0]){
            return SMALLER;
        }else{
            return EQUAL;
        }
    }
}