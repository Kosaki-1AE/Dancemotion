typedef int Bool;

#define FALSE 0
#define TRUE  1

static double abs_d(double x)
{
    return (x < 0.0) ? -x : x;
}

typedef struct {
    double input;
    double weight;
    double memory;
} Port;

typedef struct {
    double resolution;
    double state;
    double range_min;
    double range_max;
    double boundary_min;
    double boundary_max;
    double step;
} Field;

typedef enum {
    RESULT_ACCEPT = 0,
    RESULT_TRANSITION = 1,
    RESULT_REJECT = 2
} Result;

typedef enum {
    NODE_VALUE = 0,
    NODE_ADD,
    NODE_SUB,
    NODE_MUL,
    NODE_DIV
} NodeType;

typedef struct Node {
    NodeType type;
    int level;
    int branch_count;
    double responsibility;
    struct Node *parent;
    struct Node *left;
    struct Node *right;
} Node;

static double ln2(void)
{
    double x = 1.0 / 3.0;
    double x2 = x * x;
    double term = x;
    double sum = 0.0;
    int n;

    for (n = 1; n < 50; n += 2) {
        sum += term / n;
        term *= x2;
    }

    return 2.0 * sum;
}

static double responsibility_step(double a, double b)
{
    return a * b * ln2();
}

static double responsibility(Node *node)
{
    if (
        node == 0 ||
        node->parent == 0
    ) {
        return 0.0;
    }

    return responsibility(node->parent) + 1.0;
}

static void connect_child(Node *parent, Node *child, Bool right_side)
{
    if (parent == 0 || child == 0) return;

    child->parent = parent;
    child->level = parent->level + 1;

    if (right_side == TRUE) {
        parent->right = child;
    } else {
        parent->left = child;
    }

    parent->branch_count += 1;
}

static Port convert_port(double delta_z, double responsibility, double tau)
{
    Port port;
    port.input = delta_z;
    port.weight = responsibility;
    port.memory = tau;
    return port;
}

static Field update_field(Field current, Port port, double dt)
{
    Field next = current;
    double weighted_input = port.input * port.weight;

    next.state =
        current.state +
        (weighted_input - current.state) / port.memory * dt;

    return next;
}

static Result judge_identity(Field field, Bool *identity)
{
    double x = field.state;
    double epsilon = 0.000000001;

    if (field.boundary_min < x && x < field.boundary_max) {
        *identity = TRUE;
        return RESULT_ACCEPT;
    }

    if (
        abs_d(x - field.boundary_min) < epsilon ||
        abs_d(x - field.boundary_max) < epsilon
    ) {
        *identity = FALSE;
        return RESULT_TRANSITION;
    }

    *identity = FALSE;
    return RESULT_REJECT;
}

static void init_node(Node *node, NodeType type)
{
    node->type = type;
    node->level = 0;
    node->branch_count = 0;
    node->responsibility = 0.0;
    node->parent = 0;
    node->left = 0;
    node->right = 0;
}

typedef struct {
    double before;
    double after;
    Bool identity;
    Result result;
    double range_min;
    double range_max;
    double boundary_min;
    double boundary_max;
} Meaning;

typedef enum {
    ACTION_PRESERVE = 0,
    ACTION_REFRAME = 1,
    ACTION_REJECT = 2
} Action;

typedef struct {
    double target;
    double loss;
    double threshold;
} Global;

typedef struct {
    Field next_field;
    Meaning meaning;
    Global global;
    Bool identity;
    Result result;
    Action action;
} Evaluation;

static Meaning evaluate_meaning(
    Field current,
    Field next,
    Bool identity,
    Result result
)
{
    Meaning meaning;

    meaning.before = current.state;
    meaning.after = next.state;
    meaning.identity = identity;
    meaning.result = result;
    meaning.range_min = current.range_min;
    meaning.range_max = current.range_max;
    meaning.boundary_min = current.boundary_min;
    meaning.boundary_max = current.boundary_max;

    return meaning;
}

static double calculate_loss(Meaning meaning, double target)
{
    return abs_d(meaning.after - target);
}

static Global evaluate_global(
    Meaning meaning,
    double target,
    double threshold
)
{
    Global global;

    global.target = target;
    global.loss = calculate_loss(meaning, target);
    global.threshold = threshold;

    return global;
}

static Action decide_action(Result result, Global global)
{
    if (result == RESULT_REJECT) return ACTION_REJECT;
    if (global.loss <= global.threshold) return ACTION_PRESERVE;
    return ACTION_REFRAME;
}

static Evaluation import_port(
    Field current,
    Node *ast_root,
    double delta_z,
    double a,
    double b,
    double tau,
    double dt,
    double global_target,
    double global_threshold
)
{
    Evaluation evaluation;
    Port port;
    double responsibility;

    evaluation.next_field = current;
    evaluation.identity = FALSE;
    evaluation.result = RESULT_REJECT;
    evaluation.action = ACTION_REJECT;

    evaluation.meaning.before = current.state;
    evaluation.meaning.after = current.state;
    evaluation.meaning.identity = FALSE;
    evaluation.meaning.result = RESULT_REJECT;
    evaluation.meaning.range_min = current.range_min;
    evaluation.meaning.range_max = current.range_max;
    evaluation.meaning.boundary_min = current.boundary_min;
    evaluation.meaning.boundary_max = current.boundary_max;

    evaluation.global.target = global_target;
    evaluation.global.loss = 0.0;
    evaluation.global.threshold = global_threshold;

    if (tau <= 0.0) {
        return evaluation;
    }

    port = convert_port(
        delta_z,
        responsibility,
        tau
    );

    evaluation.next_field = update_field(
        current,
        port,
        dt
    );

    evaluation.result = judge_identity(
        evaluation.next_field,
        &evaluation.identity
    );

    evaluation.meaning = evaluate_meaning(
        current,
        evaluation.next_field,
        evaluation.identity,
        evaluation.result
    );

    evaluation.global = evaluate_global(
        evaluation.meaning,
        global_target,
        global_threshold
    );

    evaluation.action = decide_action(
        evaluation.result,
        evaluation.global
    );

    return evaluation;
}

int main(void)
{
    Field field;
    Evaluation evaluation;

    Node root_mul;
    Node add;
    Node value_a;
    Node value_b;
    Node value_c;

    init_node(&root_mul, NODE_MUL);
    init_node(&add, NODE_ADD);
    init_node(&value_a, NODE_VALUE);
    init_node(&value_b, NODE_VALUE);
    init_node(&value_c, NODE_VALUE);

    connect_child(&root_mul, &add, FALSE);
    connect_child(&root_mul, &value_c, TRUE);
    connect_child(&add, &value_a, FALSE);
    connect_child(&add, &value_b, TRUE);

    field.resolution = 1.0;
    field.state = 0.0;
    field.range_min = -0.5;
    field.range_max = 0.5;
    field.boundary_min = -0.5;
    field.boundary_max = 0.5;
    field.step = 0.1;

    evaluation = import_port(
        field,
        &root_mul,
        0.20,
        1.0,
        1.0,
        1.0,
        0.1,
        0.0,
        0.10
    );

    return (int)evaluation.action;
}
