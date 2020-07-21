using JuMP
using Gurobi
using LinearAlgebra

GUROBI_ENV = Gurobi.Env()

function get_empty_model()
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV)))
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

"""Generates a master problem model
# Arguments:
- `Patterns::Matrix`: matrix of cut patterns
- `demands::List`: list of products demands
"""
function master_problem(Patterns, demands)
    model = get_empty_model()

    # how many times is pattern h used?
    @variable(model, z[1:size(Patterns)[1]] >=0 , Int)

    # demand constraint
    for j in 1:length(demands)
        @constraint(model, sum(Patterns[h,j] * z[h] for h in 1:size(Patterns)[1]) >= demands[j])
    end

    # minimize number of used patterns
    @objective(model, Min, sum(z))

    return model
end


"""Generates a reduced master problem model
# Arguments:
- `Patterns::Matrix`: matrix of cut patterns
- `demands::List`: list of products demands
"""
function relaxed_master_problem(Patterns, demands)
    model = get_empty_model()

    # how many times is pattern h used?
    @variable(model, z[1:size(Patterns)[1]] >= 0)

    @constraint(
        model,
        dc[j=1:length(demands)],
        sum(Patterns[h,j] * z[h] for h in 1:size(Patterns)[1]) >= demands[j]
    )

    # minimize number of used patterns
    @objective(model, Min, sum(z))

    return model
end


""" Generates pricing problem model
# Arguments:
- `cuttings_size::Array`: array of the sizes of each cutting
- `π::Array`: Array of reduced cost coefficients
- `max_roll_length::Int`: size of raw roll
"""
function pricing_problem(cuttings_size, π, max_roll_length)
    model = get_empty_model()

    @variable(model, cuttings_in_pattern[1:length(cuttings_size)] >= 0, Int)

    @constraint(model, sum(cuttings_size[j] * cuttings_in_pattern[j] for j in 1:length(π)) <= max_roll_length)

    @objective(model, Max, sum(cuttings_in_pattern[j] * π[j] for j in 1:length(π)))

    return model
end

function main()
    MAX_ROLL_LENGTH = 100
    demands = [6,11,17,35,21, 12, 3, 46, 15, 30, 7, 4]
    demands_length = [70,50,25,15,8,20, 10, 5, 7,3,1, 23]

    Patterns = [
        1 0 0 0 0 0 0 0 0 0 0 0
        0 2 0 0 0 0 0 0 0 0 0 0
        0 0 4 0 0 0 0 0 0 0 0 0
        0 0 0 6 0 0 0 0 0 0 0 0
        0 0 0 0 12 0 0 0 0 0 0 0
        0 0 0 0 0 5 0 0 0 0 0 0
        0 0 0 0 0 0 10 0 0 0 0 0
        0 0 0 0 0 0 0 20 0 0 0 0
        0 0 0 0 0 0 0 0 14 0 0 0
        0 0 0 0 0 0 0 0 0 30 0 0
        0 0 0 0 0 0 0 0 0 0 100 0
        0 0 0 0 0 0 0 0 0 0 0 4
    ]

    iteration=1
    can_improve = true
    while can_improve
        println("iteration $iteration")
        MP = relaxed_master_problem(Patterns, demands)

        optimize!(MP)

        π = JuMP.dual.(MP[:dc])
        PP = pricing_problem(demands_length, π, MAX_ROLL_LENGTH)

        optimize!(PP)

        new_pattern = value.(PP[:cuttings_in_pattern])

        improvement = 1- dot(π, new_pattern)
        if improvement >= 0
            can_improve=false
            if termination_status(MP) == MOI.OPTIMAL
                println("Optimal relaxed solution found")
            else
                println("no solution found ", termination_status(MP))
            end
        else
            println("Current solution can improve. Adding new pattern")
            Patterns = vcat(Patterns, new_pattern')
            iteration +=1
        end
    end

    integer_MP = master_problem(Patterns, demands)
    optimize!(integer_MP)
    chosen_patterns = value.(integer_MP[:z])

    println(chosen_patterns)

    result=zeros(length(demands))
    for row_idx in 1:size(Patterns,1)
        if chosen_patterns[row_idx] > 0
            for times in 1:chosen_patterns[row_idx]
                for col_idx in 1:size(Patterns,2)
                    result[col_idx] += Patterns[row_idx,col_idx]
                end
            end
        end
    end
    println(result)

end

main()
