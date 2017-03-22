class IrisInstance
  attr_accessor :sepal_length, :sepal_width, :petal_length, :petal_width, :label, :classified_class, :clustered_class

  def initialize(sepal_length, sepal_width, petal_length, petal_width, label, classified_class = nil, clustered_class = nil)
    self.sepal_length = sepal_length
    self.sepal_width  = sepal_width
    self.petal_length = petal_length
    self.petal_width  = petal_width
    @label            = label
    @classified_class = classified_class
    @clustered_class  = clustered_class
  end

  def classified_class_mismatched?
    if classified_class.nil? # label not given, or not classified yet
      false
    else
      label != classified_class
    end
  end

  def label
    @label.nil? ? "Unlabeled" : @label
  end

  def classified_class
    @classified_class.nil? ? "Unclassified" : @classified_class
  end

  def clustered_class
    @clustered_class.nil? ? "Unclustered" : @clustered_class
  end

  def to_s
    "[#{sepal_length}, #{sepal_width}, #{petal_length}, #{petal_width}, #{label}, #{classified_class}, #{clustered_class}]"
  end

  def clone
    IrisInstance.new(sepal_length, sepal_width, petal_length, petal_width, @label, @classified_class, @clustered_class)
  end
end
